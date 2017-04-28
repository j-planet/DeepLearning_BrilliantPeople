import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf

from layers.abstract_layer import AbstractLayer


class FullyConnectedLayer(AbstractLayer):

    def __init__(self, input_, weightDim, activation=None, loggerFactory=None):
        """
        :type weightDim: tuple
        """

        self.weightDim = weightDim
        self.input = input_


        super().__init__(activation, loggerFactory)


    def make_graph(self):
        self.weights = tf.Variable(tf.random_normal(self.weightDim), name='weights')
        self.biases = tf.Variable(tf.random_normal([self.weightDim[-1]]), name='biases')

        self.output = tf.matmul(self.input, self.weights) + self.biases

    def output_shape(self, inputDim_):
        return inputDim_[0], self.weightDim[-1]


if __name__ == '__main__':

    v = tf.Variable(tf.random_normal([2, 5]))
    layer = FullyConnectedLayer(v, (5, 2), activation='relu')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v))
    print('\n-------- OUTPUT --------')
    print(output)
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)