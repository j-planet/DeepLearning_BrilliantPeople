import glob, os
import tensorflow as tf
import logging


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


def setup_logging(logFilename_, level_=logging.DEBUG):
    logging.basicConfig(level=level_,
                        format='%(asctime)s %(name)-15s %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(logFilename_, encoding='utf8'),
                                  logging.StreamHandler()])
