import glob, os
import tensorflow as tf


def tensorflowFilewriter(writerDir):

    # clear existing logs first
    for f in glob.glob(writerDir.strip('/') + '/*'):
        os.remove(f)

    # for f in glob.glob('./logs/word2vec/validation/*'): os.remove(f)
    train_writer = tf.summary.FileWriter(writerDir)
    # valid_writer = tf.summary.FileWriter('./logs/word2vec/validation')

    return train_writer


def reshape_x_for_non_dynamic(x_, numSeqs_, seqLen_):
    # reshape stuff
    # input shape: (batch_size x n_steps x n_input)
    # output shape: n_steps tensors, each of shape (batch_size x n_input)
    return tf.split(
        tf.reshape(
            tf.transpose(x_, [1, 0, 2]),
            [-1, seqLen_]),
        numSeqs_, 0)