import tensorflow as tf

from layers.abstract_layer import AbstractLayer


class LocalNormLayer(AbstractLayer):
    """
    The most vanilla version of local norm -- norm by std
    """

    def __init__(self, input_, inputDim_,
                 depth_radius=None, bias=None,
                 activation=None, loggerFactory_=None):

        assert len(inputDim_) == 4

        self.depth_radius = depth_radius
        self.bias = bias
        # self.alpha = alpha
        # self.beta = beta

        super().__init__(input_, inputDim_, activation, loggerFactory=loggerFactory_)


    def make_graph(self):
        self.output = tf.nn.local_response_normalization(self.input, self.depth_radius, self.bias)

    @property
    def output_shape(self):
        return self.inputDim

    @classmethod
    def new(cls, depth_radius=None, bias=None, activation=None):
        return lambda i, iDim, loggerFactory=None: cls(i, iDim, depth_radius, bias, activation, loggerFactory)


if __name__ == '__main__':


    inputDim = (1, 1, 1, 3)
    v = tf.Variable(tf.random_normal(inputDim))
    maker = LocalNormLayer.new(depth_radius=5, bias=0)
    layer = maker(v, inputDim)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(layer.output)
    print('-------- INPUT --------')
    print(sess.run(v)[0, 0,:,:])
    print('\n-------- OUTPUT --------')
    print(output[0, 0, :,:])
    print('\n-------- OUTPUT SHAPE --------')
    print(output.shape)
    print(layer.output_shape)