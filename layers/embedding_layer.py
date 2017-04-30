import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow.contrib.learn import preprocessing

from layers.abstract_layer import AbstractLayer
from data_readers import DataReader_Text


class EmbeddingLayer(AbstractLayer):

    def __init__(self, input_, inputDim_,
                 vocabSize_, embeddingDim,
                 activation=None, loggerFactory=None):
        """
        :type vocabSize_: int
        :type embeddingDim: int
        """

        assert vocabSize_ > 0
        assert embeddingDim >0

        self.vocabSize = vocabSize_
        self.embeddingDim = embeddingDim

        super().__init__(input_, inputDim_, activation, loggerFactory)

    def make_graph(self):
        self.W = tf.Variable(tf.random_uniform([self.vocabSize, self.embeddingDim], -1.0, 1.0), name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input)

    @property
    def output_shape(self):
        return self.inputDim

    @classmethod
    def new(cls, keepProb, activation=None):
        return lambda input_, inputDim_, loggerFactory=None: cls(input_, inputDim_, keepProb, activation, loggerFactory)

if __name__ == '__main__':
    dr = DataReader_Text('../data/peopleData/earlyLifeTokensFile_polsci.json', 'bucketing', 5, 1)

    vocabProcessor = preprocessing.VocabularyProcessor(dr.maxXLen)
    x = vocabProcessor.fit_transform()


    inputShape = [2, 5]
    v = tf.Variable(tf.random_normal(inputShape))
    maker = EmbeddingLayer.new(0.5)
    layer = maker(v, inputShape)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v))
    print('\n-------- OUTPUT --------')
    print(output)
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)
    print(layer.output_shape)