import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow import name_scope


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


if __name__ == '__main__':
    v = tf.Variable(tf.random_normal([1, 2, 5, 1]))
    layer = MaxpoolLayer(v, (2, 2), (1,1), padding='VALID')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v)[0,:,:,0])
    print('\n-------- OUTPUT --------')
    print(output[0,:,:,0])
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)