import glob, os
import tensorflow as tf


def tensorflowFilewriter(writerDir):

    # clear existing logs first
    for f in glob.glob(writerDir):
        os.remove(f)

    # for f in glob.glob('./logs/word2vec/validation/*'): os.remove(f)
    train_writer = tf.summary.FileWriter(writerDir)
    # valid_writer = tf.summary.FileWriter('./logs/word2vec/validation')

    return train_writer