import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf

from layers.abstract_layer import AbstractLayer


class DropoutLayer(AbstractLayer):

    def __init__(self, input_, inputDim_,
                 keepProb,
                 activation=None, loggerFactory=None):
        """
        :type keepProb: float
        """

        assert 0. < keepProb <= 1
        self.keepProb = keepProb

        super().__init__(input_, inputDim_, activation, loggerFactory)

        self.print('dropout keep prob: %0.3f' % keepProb)

    def make_graph(self):
        self.output = tf.nn.dropout(self.input, self.keepProb)

    @property
    def output_shape(self):
        return self.inputDim

    @classmethod
    def new(cls, keepProb, activation=None):
        return lambda input_, inputDim_, loggerFactory=None: cls(input_, inputDim_, keepProb, activation, loggerFactory)

if __name__ == '__main__':
    inputShape = [2, 5]
    v = tf.Variable(tf.random_normal(inputShape))
    maker = DropoutLayer.new(0.5)
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