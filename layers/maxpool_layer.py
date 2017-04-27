import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow import name_scope

from utilities import filter_output_size


# ------ stack of LSTM - bi-directional RNN layer ------
class MaxpoolLayer(object):

    def __init__(self, input_, ksize, strides=(1,1), padding='VALID', loggerFactory=None):
        """
        :type ksize: tuple
        :type strides: tuple
        """

        assert len(ksize) == len(strides) == 2, 'We only maxpool in the 2nd and 3rd dimensions.'

        self.ksize = [1, *ksize, 1]
        self.strides = [1, *strides, 1]
        self.padding = padding
        self.print = print if loggerFactory is None else loggerFactory.getLogger('Layer').info
        self.print('Constructing: ' + self.__class__.__name__)

        with name_scope(self.__class__.__name__):

            self.output = tf.nn.max_pool( input_,
                ksize=self.ksize, strides=self.strides,
                padding=self.padding,
                name='pool')

    def output_size(self, inputDim):
        """
        :type inputDim: tuple 
        """

        assert len(inputDim) == 4

        return inputDim[0], \
               filter_output_size(inputDim[1], self.ksize[0], self.strides[0], self.padding), \
               filter_output_size(inputDim[2], self.ksize[1], self.strides[1], self.padding), \
               inputDim[3]


if __name__ == '__main__':
    inputShape = [1, 30, 30, 2]
    ksize = (2, 7)
    stride = (3, 4)
    v = tf.Variable(tf.random_normal(inputShape))
    l1 = MaxpoolLayer(v, ksize, stride, padding='SAME')
    l2 = MaxpoolLayer(v, ksize, stride, padding='VALID')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output1 = sess.run(l1.output)
    output2 = sess.run(l2.output)

    print('-------- INPUT --------')
    print(sess.run(v)[0,:,:,:])
    print('-------- INPUT SHAPE --------')
    print(inputShape)

    print('\n-------- OUTPUT (SAME) --------')
    print(output1[0,:,:,0])
    print('\n-------- OUTPUT SHAPE (SAME) --------')
    print(output1.shape)
    #
    print('\n\n-------- OUTPUT (VALID) --------')
    print(output2[0,:,:,0])
    print('\n-------- OUTPUT SHAPE (VALID) --------')
    print(output2.shape)