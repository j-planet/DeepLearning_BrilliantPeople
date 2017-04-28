import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf

from layers.abstract_layer import AbstractLayer


# ------ stack of LSTM - bi-directional RNN layer ------
class MaxpoolLayer(AbstractLayer):

    def __init__(self, input_, inputDim_, ksize, strides=(1,1), padding='VALID', activation=None, loggerFactory=None):
        """
        :type ksize: tuple
        :type strides: tuple
        """

        assert len(ksize) == len(strides) == 2, 'We only maxpool in the 2nd and 3rd dimensions.'
        assert len(inputDim_) == 4

        self.ksize = [1, *ksize, 1]
        self.strides = [1, *strides, 1]
        self.padding = padding

        super().__init__(input_, inputDim_, activation, loggerFactory)

    def make_graph(self):
        self.output = tf.nn.max_pool(self.input,
                                     ksize=self.ksize, strides=self.strides, padding=self.padding,
                                     name='pool')
    @property
    def output_shape(self):

        # return (self.inputDim[0],
        #         *[filter_output_size(self.inputDim[i], self.ksize[i], self.strides[i], self.padding) for i in [1,2]],
        #         self.inputDim[3])

        return self.inputDim

    @classmethod
    def new(cls, ksize, strides=(1, 1), padding='VALID', activation=None):
        return lambda input_, inputDim_, loggerFactory=None: \
            cls(input_, inputDim_,
                ksize, strides, padding, activation, loggerFactory)

if __name__ == '__main__':
    inputShape = [1, 3, 2, 1]
    ksize = (100, 100)
    stride = (1, 1)
    v = tf.Variable(tf.random_normal(inputShape))

    maker1 = MaxpoolLayer.new(ksize, stride, padding='SAME')
    maker2 = MaxpoolLayer.new(ksize, stride, padding='VALID')
    l1 = maker1(v, inputShape)
    l2 = maker1(v, inputShape)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output1 = sess.run(l1.output)
    output2 = sess.run(l2.output)

    print('-------- INPUT --------')
    print(sess.run(v)[0,:,:,0])
    print('-------- INPUT SHAPE --------')
    print(inputShape)

    print('\n-------- OUTPUT (SAME) --------')
    print(output1[0,:,:,0])
    print('\n-------- OUTPUT SHAPE (SAME) --------')
    print(output1.shape)
    print(l1.output_shape)

    print('\n\n-------- OUTPUT (VALID) --------')
    print(output2[0,:,:,0])
    print('\n-------- OUTPUT SHAPE (VALID) --------')
    print(output2.shape)
    print(l2.output_shape)