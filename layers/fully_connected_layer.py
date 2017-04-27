import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow import name_scope

from utilities import str_2_activation_function



class FullyConnectedLayer(object):

    def __init__(self, input_, weightDim, activation=None, loggerFactory=None):
        """
        :type weightDim: tuple
        """

        self.weightDim = weightDim
        self.activationFunc = str_2_activation_function(activation)
        self.print = print if loggerFactory is None else loggerFactory.getLogger('Model').info
        self.print('Constructing: ' + self.__class__.__name__)

        with name_scope(self.__class__.__name__):
            self.weights = tf.Variable(tf.random_normal(weightDim), name='weights')
            self.biases = tf.Variable(tf.random_normal([weightDim[-1]]), name='biases')

            self.output = tf.matmul(input_, self.weights) + self.biases

    def output_size(self, batchSize):
        return batchSize, self.weightDim[-1]

if __name__ == '__main__':

    v = tf.Variable(tf.random_normal([2, 5]))
    layer = FullyConnectedLayer(v, 2, activation='relu')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v))
    print('\n-------- OUTPUT --------')
    print(output)
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)