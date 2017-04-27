import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow import name_scope

from utilities import str_2_activation_function, filter_output_size


class ConvLayer(object):

    def __init__(self, input_,
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

        self.filterShape = filterShape
        self.numFeaturesPerFilter = numFeaturesPerFilter
        self.strides = strides
        self.keepProb = keepProb
        self.padding = padding
        self.activationFunc = str_2_activation_function(activation)

        self.print = print if loggerFactory is None else loggerFactory.getLogger('Model').info
        self.print('Constructing: ' + self.__class__.__name__)

        input_ = tf.expand_dims(input_, -1)  # expects 3-d input of shape batch size x num sequences x vec dim

        with name_scope(self.__class__.__name__):

            filterMat = tf.Variable(tf.truncated_normal([*filterShape, 1, self.numFeaturesPerFilter], stddev=0.1), name='W')
            filterBiases = tf.Variable(tf.constant(0.1, shape=[self.numFeaturesPerFilter]), name='b')

            self.conv = tf.nn.conv2d(input_, filterMat, strides=[1, *self.strides, 1], padding=self.padding, name='conv')    # supports only 'VALID' for now
            self.activated = self.activationFunc(tf.nn.bias_add(self.conv, filterBiases))

            self.output = tf.nn.dropout(self.activated, self.keepProb)

    def output_size(self, inputDim):
        """
        :type inputDim: tuple 
        """

        assert len(inputDim) == 3

        return inputDim[0], \
               filter_output_size(inputDim[1], self.filterShape[0], self.strides[0], self.padding), \
               filter_output_size(inputDim[2], self.filterShape[1], self.strides[1], self.padding), \
               self.numFeaturesPerFilter

if __name__ == '__main__':

    v = tf.Variable(tf.random_normal([1, 3, 5]))
    layer = ConvLayer(v, (2, 1), 4, (2, 3), activation='relu')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v)[0,:,:])
    print('\n-------- OUTPUT --------')
    print(output[0,:,:])
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)