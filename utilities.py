from datetime import datetime
import glob, os, logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pprint import pformat
import multiprocessing



def dir_create_n_clear(*pathComponents):
    res = os.path.join(*pathComponents)

    if not os.path.exists(res):
        os.mkdir(res)
    else:
        for f in glob.glob(os.path.join(res, '*')):
            os.remove(f)

    return res


def tensorflowFilewriters(writerDir):
    """ 
    :return: trainWriter, ValidateWriter
    """

    return tf.summary.FileWriter(dir_create_n_clear(writerDir, 'train')), \
           tf.summary.FileWriter(dir_create_n_clear(writerDir, 'validation'))


def reshape_x_for_non_dynamic(x_, numSeqs_, seqLen_):
    # reshape stuff
    # input shape: (batch_size x numSeqs_ x seqLen_)
    # output shape: numSeqs_ tensors, each of shape (batch_size x seqLen_)
    return tf.split(
        tf.reshape(
            tf.transpose(x_, [1, 0, 2]),
            [-1, seqLen_]),
        numSeqs_, 0)


def label_comparison(trueYInds_, predYInds_, names_, yLabelTexts_, logFunc_, verbose_ = False):
    logFunc_ = logFunc_ or print

    if verbose_:
        logFunc_('True label became... --> ?')

    wrongCount = {} # { (true, pred): count }

    for i, name in enumerate(names_):

        trueLabel = yLabelTexts_[trueYInds_[i]]
        predLabel = yLabelTexts_[predYInds_[i]]

        correct = trueLabel == predLabel

        if not correct:
            wrongCount[(trueLabel, predLabel)] = wrongCount.get((trueLabel, predLabel), 0) + 1

        if verbose_:
            logFunc_('%-20s %s --> %s %s' % (name, trueLabel, predLabel, '' if correct else '(wrong)'))

    logFunc_('Wrong prediction count for %d samples:' % len(names_))
    logFunc_(pformat(wrongCount))


def save_matrix_img(mats_, title, outputDir_, transpose_=False):

    d = np.array(mats_) if len(mats_[0].shape) == 1 else np.concatenate(mats_, axis=1)

    fig = plt.figure()
    ax = plt.subplot(111)
    heatmap = ax.matshow(np.transpose(d) if transpose_ else d, cmap='gray')
    plt.colorbar(heatmap)
    plt.title(title)
    fig.savefig(os.path.join(outputDir_, title+'.png'))


def setup_logging(logFilename_, level_=logging.DEBUG):
    print('setup_logging for', logFilename_)
    logger = logging.getLogger()

    if len(logger.handlers) > 0:
        for hdl in logger.handlers:
            hdl.stream.close()
            logger.removeHandler(hdl)

        file_handler = logging.FileHandler(logFilename_)

        file_handler.setLevel(level_)
        formatter = logging.Formatter('%(asctime)s %(name)-15s %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.basicConfig(level=level_,
                        format='%(asctime)s %(name)-15s %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(logFilename_, encoding='utf8'),
                                  logging.StreamHandler()])

def create_time_dir(baseDir):
    res = os.path.join(baseDir, datetime.now().strftime('%m%d%Y %H:%M:%S'))
    if not os.path.exists(res): os.makedirs(res, exist_ok=True)

    return res


class LoggerFactory(object):

    def __init__(self, outputDir_):

        self.filename = os.path.join(outputDir_, 'log.log')
        setup_logging(self.filename)
        logging.info('Logging to ' + self.filename)

        self.loggers = {}

    def getLogger(self, n_):

        if n_ in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[n_]

        self.loggers[n_] = logging.getLogger(n_)
        return self.loggers[n_]


def last_relevant(output_, lengths_, numRows_=1):
    batch_size = tf.shape(output_)[0]
    max_length = tf.shape(output_)[1]
    out_size = int(output_.get_shape()[2])
    index = tf.expand_dims(tf.range(0, batch_size),-1) * max_length \
            + tf.tile(tf.expand_dims(lengths_ - 1, -1), [1, numRows_]) + tf.range(-numRows_+1, 1)

    flat = tf.reshape(output_, [-1, out_size])

    return tf.gather(flat, index)


def str_2_activation_function(name):
    assert name in [None, 'relu', 'sigmoid', 'tanh']

    if name is None: return lambda x: x
    if name=='relu': return tf.nn.relu
    if name=='tanh': return tf.nn.tanh
    if name=='sigmoid': return tf.nn.sigmoid


def filter_output_size(inputLen, filterWidth, stride, padding):
    assert min(inputLen, filterWidth, stride) > 0
    assert padding in ['VALID', 'SAME']

    if padding == 'VALID':
        assert filterWidth <= inputLen
        assert stride <= inputLen

        return int((inputLen - filterWidth)/stride) + 1

    if padding == 'SAME':

        if filterWidth > inputLen or stride > inputLen:
            return 1

        return int(np.ceil(inputLen / stride))

def run_with_processor(trainFunc, useCPU):
    """
    :param trainFunc: lambda sess: train(sess, ...)
    """


    tf.reset_default_graph()

    if useCPU:

        numCores = multiprocessing.cpu_count() - 1
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=numCores,
                                inter_op_parallelism_threads=numCores)

        with tf.device('/cpu:0'):
            sess = tf.InteractiveSession(config=config)
            return trainFunc(sess)
    else:
        sess = tf.InteractiveSession(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85),
                                  allow_soft_placement=True))
        return trainFunc(sess)

def make_params_dict(paramsKeyValuesList):
    """
    :param paramsKeyValuesList:  [(name, list of values), ...]
    :return: a list of dictionaries of {name: value, ...}
    """

    keys = [v[0] for v in paramsKeyValuesList]
    vals = [v[1] for v in paramsKeyValuesList]

    return [dict(zip(keys, params)) for params in itertools.product(*vals)]


def convert_to_2d(t, d):
    assert len(d) > 2

    newSecondD = np.product(d[1:])
    return tf.reshape(t, [-1, newSecondD]), newSecondD

def convert_to_3d(t, d):
    assert len(d) > 3

    newThirdD = np.product(d[2:])
    return tf.reshape(t, [-1, d[1], newThirdD]), newThirdD