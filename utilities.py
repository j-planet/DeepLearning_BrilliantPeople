from datetime import datetime
import glob, os, logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


def label_comparison(trueYInds_, predYInds_, names_, yLabelTexts_, logFunc_):
    logFunc_ = logFunc_ or print
    logFunc_('True label became... --> ?')

    for i, name in enumerate(names_):
        logFunc_('%-20s %s --> %s %s' % (name, yLabelTexts_[trueYInds_[i]], yLabelTexts_[predYInds_[i]],
                                         '(wrong)' if trueYInds_[i] != predYInds_[i] else ''))


def save_matrix_img(mats_, title, outputDir_, transpose_=False):

    d = np.array(mats_) if len(mats_[0].shape) == 1 else np.concatenate(mats_, axis=1)

    fig = plt.figure()
    ax = plt.subplot(111)
    heatmap = ax.matshow(np.transpose(d) if transpose_ else d, cmap='gray')
    plt.colorbar(heatmap)
    plt.title(title)
    fig.savefig(os.path.join(outputDir_, title+'.png'))


def setup_logging(logFilename_, level_=logging.DEBUG):
    logging.basicConfig(level=level_,
                        format='%(asctime)s %(name)-15s %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(logFilename_, encoding='utf8'),
                                  logging.StreamHandler()])

def create_time_dir(baseDir):
    res = os.path.join(baseDir, datetime.now().strftime('%m%d%Y %H:%M:%S'))
    if not os.path.exists(res): os.mkdir(res)

    return res


class LoggerFactory(object):

    def __init__(self, outputDir_):

        self.filename = os.path.join(outputDir_, 'log.log')
        setup_logging(self.filename)
        logging.info('Logging to ' + self.filename)

        self.loggers = {}

    def getLogger(self, n_):
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