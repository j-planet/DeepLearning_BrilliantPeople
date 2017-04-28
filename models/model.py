import tensorflow as tf

from models.abstract_model import AbstractModel
from data_reader import DataReader
from layers.abstract_layer import AbstractLayer
from layers.rnn_layer import RNNLayer
from layers.conv_layer import ConvLayer
from layers.maxpool_layer import MaxpoolLayer
from layers.fully_connected_layer import FullyConnectedLayer


class ConvMaxpoolLayer(AbstractLayer):

    def __init__(self, input_, inputDim_, convParams_, maxPoolParams_, activation=None, loggerFactory=None):
        self.convParams = convParams_
        self.maxPoolParams = maxPoolParams_

        super().__init__(input_, inputDim_, activation, loggerFactory)

    def make_graph(self):
        self.convLayer = ConvLayer(self.input,
                                   self.inputDim,
                                   **self.convParams)

        self.maxPoolLayer = MaxpoolLayer(self.convLayer.output,
                                         self.convLayer.output_shape,
                                         **self.maxPoolParams)

        self.output = self.maxPoolLayer.output

    @property
    def output_shape(self):
        return self.maxPoolLayer.output_shape

    @classmethod
    def new(cls, convParams_, maxPoolParams_, activation=None, loggerFactory=None):
        return lambda input_, inputDim_: cls(input_, inputDim_,
                                             convParams_, maxPoolParams_, activation, loggerFactory)


class Model(AbstractModel):

    def __init__(self, input_, loggerFactory_=None):
        self.l2RegLambda = 0
        self.initialLearningRate = 1e-3

        super().__init__(input_, loggerFactory_)

    def make_graph(self):

        self.add_layer(
            RNNLayer.new(
                [16], numStepsToOutput_ = 3), self.input, (-1, -1, self.vecDim))

        self.add_layer(
            ConvMaxpoolLayer.new(
                convParams_={'filterShape': (1,1), 'numFeaturesPerFilter': 8, 'activation': None},
                maxPoolParams_={'ksize': (1,1)}))

        self.add_layer(
            FullyConnectedLayer.new(self.numClasses))


if __name__ == '__main__':

    datadir = '../data/peopleData/2_samples'
    # datadir = '../data/peopleData/earlyLifesWordMats/politician_scientist'

    batchSize = 20
    dr = DataReader(datadir, 'bucketing', 20)
    model = Model(dr.input)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(24):
        fd = dr.get_next_training_batch()[0]
        _, c, acc = model.train_op(sess, fd, True)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))