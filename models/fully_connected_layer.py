import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow import name_scope

from utilities import str_2_activation_function



class FullyConnectedLayer(object):

    def __init__(self, input_, numClasses, activationFuncName=None, loggerFactory=None):
        """
        :type numClasses: int
        """

        self.numClasses = numClasses
        self.activationFunc = str_2_activation_function(activationFuncName)
        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('Model').info
        self._logFunc('Constructing: ' + self.__class__.__name__)

        with name_scope(self.__class__.__name__):
            self.weights = tf.Variable(tf.random_normal([input_.get_shape()[-1].value, self.numClasses]), name='weights')
            self.biases = tf.Variable(tf.random_normal([self.numClasses]), name='biases')

            self.output = tf.matmul(input_, self.weights) + self.biases


if __name__ == '__main__':

    v = tf.Variable(tf.random_normal([2, 5]))
    layer = FullyConnectedLayer(v, 2, activationFuncName='relu')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v))
    print('\n-------- OUTPUT --------')
    print(output)
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)