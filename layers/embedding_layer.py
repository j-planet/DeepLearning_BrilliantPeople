import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf

from layers.abstract_layer import AbstractLayer


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

        super().__init__(input_, inputDim_, activation, loggerFactory)

        self.print('dropout keep prob: %0.3f' % keepProb)

    def make_graph(self):
        self.W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    @property
    def output_shape(self):
        return self.inputDim

    @classmethod
    def new(cls, keepProb, activation=None):
        return lambda input_, inputDim_, loggerFactory=None: cls(input_, inputDim_, keepProb, activation, loggerFactory)

if __name__ == '__main__':
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