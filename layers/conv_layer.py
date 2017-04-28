import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf

from utilities import filter_output_size
from layers.abstract_layer import AbstractLayer


class ConvLayer(AbstractLayer):

    def __init__(self, input_, inputDim_,
                 filterShape, numFeaturesPerFilter, strides=(1,1), padding='VALID', activation=None,
                 keepProb=1, loggerFactory=None):
        """
        :type filterShape: tuple
        :type numFeaturesPerFilter: int
        :type keepProb: float
        :type strides: tuple
        :type activation: str
        """

        assert len(filterShape) == len(strides) == 2, 'We only conv in the 2nd and 3rd dimensions.'
        assert len(inputDim_) in [3, 4]

        self.filterShape = filterShape
        self.numFeaturesPerFilter = numFeaturesPerFilter
        self.strides = strides
        self.keepProb = keepProb
        self.padding = padding

        super().__init__(input_, inputDim_, activation, loggerFactory)

    def input_modifier(self, val):

        if len(self.inputDim) == 3:
            self.print('Expanding input dimension by 1.')
            return tf.expand_dims(val, -1)  # expects 3-d input of shape batch size x num sequences x vec dim

        return val

    def make_graph(self):
        filterMat = tf.Variable(tf.truncated_normal([*self.filterShape, 1, self.numFeaturesPerFilter], stddev=0.1), name='W')
        filterBiases = tf.Variable(tf.constant(0.1, shape=[self.numFeaturesPerFilter]), name='b')

        self.conv = tf.nn.conv2d(self.input, filterMat, strides=[1, *self.strides, 1], padding=self.padding, name='conv')    # supports only 'VALID' for now

        self.output = tf.nn.dropout(
            tf.nn.bias_add(self.conv, filterBiases),
            self.keepProb)

    @property
    def output_shape(self):

        return self.inputDim[0], \
               filter_output_size(self.inputDim[1], self.filterShape[0], self.strides[0], self.padding), \
               filter_output_size(self.inputDim[2], self.filterShape[1], self.strides[1], self.padding), \
               self.numFeaturesPerFilter

    @classmethod
    def new(cls, filterShape, numFeaturesPerFilter, strides=(1, 1), padding='VALID', activation=None,
            keepProb=1, loggerFactory=None):
        return lambda input_, inputDim_: cls(input_, inputDim_,
                                             filterShape, numFeaturesPerFilter, strides, padding, activation,
                                             keepProb, loggerFactory)


if __name__ == '__main__':

    inputDim = [1, 3, 5]
    v = tf.Variable(tf.random_normal(inputDim))
    maker = ConvLayer.new((2, 1), 4, (2, 3), activation='relu')
    layer = maker(v, inputDim)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v)[0,:,:])
    print('\n-------- OUTPUT --------')
    print(output[0,:,:])
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)
    print(layer.output_shape)