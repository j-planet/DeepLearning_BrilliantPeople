import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
import numpy as np

from layers.abstract_layer import AbstractLayer
from data_readers.text_data_reader import TextDataReader


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
        assert len(inputDim_) == 2

        self.vocabSize = vocabSize_
        self.embeddingDim = embeddingDim

        super().__init__(input_, inputDim_, activation, loggerFactory)

        self.print('vocab size: %d' % vocabSize_)
        self.print('embedding dim: %d' % embeddingDim)
        self.print('input dim: ' + str(inputDim_))

    def make_graph(self):
        self.W = tf.Variable(tf.random_uniform([self.vocabSize, self.embeddingDim], -1.0, 1.0), name="W")
        self.output = tf.nn.embedding_lookup(self.W, self.input)

    @property
    def output_shape(self):
        return self.inputDim[0], self.inputDim[1], self.embeddingDim

    @classmethod
    def new(cls, vocabSize_, embeddingDim, activation=None):
        return lambda input_, inputDim_, loggerFactory=None: \
            cls(input_, inputDim_, vocabSize_, embeddingDim, activation, loggerFactory)

if __name__ == '__main__':
    dr = TextDataReader('../data/peopleData/tokensfiles/pol_sci.json', 'bucketing', 5, 1)
    fd = dr.get_next_training_batch()[0]
    # inputVal = np.array([[4, 5, 1, 0, 0], [2, 0, 0, 0, 0]])
    # inputShape = inputVal.shape
    # v = tf.Variable(inputVal)

    maker = EmbeddingLayer.new(vocabSize_=dr.vocabSize, embeddingDim=32)
    layer = maker(dr.input['x'], [-1, dr.maxXLen])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # output = sess.run(layer.output)
    output = sess.run(layer.output, fd)

    print('-------- INPUT --------')
    # print(sess.run(v))
    print(fd[dr.input['x']])
    print('\n-------- OUTPUT --------')
    print(output)
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)
    print(layer.output_shape)