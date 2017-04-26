import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow import name_scope

from utilities import str_2_activation_function


class ConvLayer(object):

    def __init__(self, input_,
                 filterShape, numFeaturesPerFilter, strides=(1,1), padding='VALID',
                 activationFuncName=None,
                 keepProb=1, loggerFactory=None):
        """
        :type filterShape: tuple
        :type numFeaturesPerFilter: int
        :type keepProb: float
        :type strides: tuple
        :type activationFuncName: str
        """

        assert len(filterShape) == len(strides) == 2, 'We only conv in the 2nd and 3rd dimensions.'
        assert padding in ['VALID', 'SAME']

        self.filterShapes = filterShape
        self.numFeaturesPerFilter = numFeaturesPerFilter
        self.strides = strides
        self.keepProb = keepProb
        self.padding = padding
        self.activationFunc = str_2_activation_function(activationFuncName)

        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('Model').info
        self._logFunc('Constructing: ' + self.__class__.__name__)

        input_ = tf.expand_dims(input_, -1)  # expects 3-d input of shape batch size x num sequences x vec dim

        with name_scope(self.__class__.__name__):

            filterMat = tf.Variable(tf.truncated_normal([*filterShape, 1, self.numFeaturesPerFilter], stddev=0.1), name='W')
            filterBiases = tf.Variable(tf.constant(0.1, shape=[self.numFeaturesPerFilter]), name='b')

            self.conv = tf.nn.conv2d(input_, filterMat, strides=[1, *self.strides, 1], padding=self.padding, name='conv')
            self.activated = self.activationFunc(tf.nn.bias_add(self.conv, filterBiases))

            self.output = tf.nn.dropout(self.activated, self.keepProb)

if __name__ == '__main__':

    v = tf.Variable(tf.random_normal([1, 2, 3]))
    layer = ConvLayer(v, (1, 1), 1, activationFuncName='relu')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v)[0,:,:])
    print('\n-------- OUTPUT --------')
    print(output[0,:,:])
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)