import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
import numpy as np

from layers.abstract_layer import AbstractLayer


class FullyConnectedLayer(AbstractLayer):

    def __init__(self, input_, inputDim_, outputLastDimLen, activation=None, loggerFactory=None):
        """
        :type outputLastDimLen: int
        """

        self.outputLastDimLen = outputLastDimLen

        super().__init__(input_, inputDim_, activation, loggerFactory)

    def _weight_first_dim(self):
        return np.product(self.inputDim[1:])

    def make_graph(self):
        self.weightDim = (self._weight_first_dim(), self.outputLastDimLen)
        self.weights = tf.Variable(tf.random_normal(self.weightDim), name='weights')

        self.biases = tf.Variable(tf.random_normal([self.weightDim[-1]]), name='biases')

        self.output = tf.matmul(self.input, self.weights) + self.biases

    @property
    def output_shape(self):
        return self.inputDim[0], self.weightDim[-1]

    def input_modifier(self, val):
        self.print('Flattening the input into a 2D tensor.')
        return tf.reshape(val, [-1, self._weight_first_dim()])

    @classmethod
    def new(cls, outputLastDimLen, activation=None):
        return lambda input_, inputDim_, loggerFactory=None: cls(input_, inputDim_, outputLastDimLen, activation, loggerFactory)

if __name__ == '__main__':
    inputShape = [2, 5]
    v = tf.Variable(tf.random_normal(inputShape))
    maker = FullyConnectedLayer.new(2, activation='relu')
    layer = maker(v, [2, 5])

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