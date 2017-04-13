import glob, os
import tensorflow as tf


def tensorflowFilewriters(writerDir):
    """ 
    :return: trainWriter, ValidateWriter
    """

    def _make_one_writer(dir_):
        # clear existing logs first
        for f in glob.glob(os.path.join(dir_, '*')):
            os.remove(f)

        return tf.summary.FileWriter(dir_)

    return _make_one_writer(os.path.join(writerDir, 'train')), \
           _make_one_writer(os.path.join(writerDir, 'validation'))


def reshape_x_for_non_dynamic(x_, numSeqs_, seqLen_):
    # reshape stuff
    # input shape: (batch_size x n_steps x n_input)
    # output shape: n_steps tensors, each of shape (batch_size x n_input)
    return tf.split(
        tf.reshape(
            tf.transpose(x_, [1, 0, 2]),
            [-1, seqLen_]),
        numSeqs_, 0)