import tensorflow as tf

from layers.abstract_layer import AbstractLayer
from layers.conv_layer import ConvLayer
from layers.local_norm_layer import LocalNormLayer

class ConvLocalnormLayer(AbstractLayer):

    def __init__(self, input_, inputDim_, convParams_, localnormParams_={}, activation=None, loggerFactory=None):
        self.convParams = convParams_
        self.localnormParams = localnormParams_

        super().__init__(input_, inputDim_, activation, loggerFactory)

    def make_graph(self):
        self.convLayer = ConvLayer(self.input,
                                   self.inputDim,
                                   **self.convParams, loggerFactory=self.loggerFactory)

        self.localnormLayer = LocalNormLayer(self.convLayer.output,
                                             self.convLayer.output_shape,
                                             **self.localnormParams, loggerFactory_=self.loggerFactory)

        self.output = self.localnormLayer.output

    @property
    def output_shape(self):
        return self.localnormLayer.output_shape

    @classmethod
    def new(cls, convParams_, localnormParams_={}, activation=None):
        return lambda input_, inputDim_, loggerFactory=None: \
            cls(input_, inputDim_,
                convParams_, localnormParams_, activation, loggerFactory)


if __name__ == '__main__':
    inputShape = [2, 5, 3]
    v = tf.Variable(tf.random_normal(inputShape))

    maker = ConvLocalnormLayer.new(convParams_={'filterShape': (2, 2), 'numFeaturesPerFilter': 16, 'activation': 'relu'})
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