from datetime import datetime
import glob, os, logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def tensorflowFilewriters(writerDir):
    """ 
    :return: trainWriter, ValidateWriter
    """

    def _make_one_writer(dir_):

        if not os.path.exists(dir_): os.mkdir(dir_)

        # clear existing logs first
        for f in glob.glob(os.path.join(dir_, '*')):
            os.remove(f)

        return tf.summary.FileWriter(dir_)

    return _make_one_writer(os.path.join(writerDir, 'train')), \
           _make_one_writer(os.path.join(writerDir, 'validation'))


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
