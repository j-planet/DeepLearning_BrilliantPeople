import tensorflow as tf

from models.abstract_model import AbstractModel
from data_readers.embedding_data_reader import EmbeddingDataReader
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.dropout_layer import DropoutLayer

from utilities import make_params_dict


class RNNConfig(object):
    def __init__(self, numCellUnits, keepProbs, activation=None):

        assert len(numCellUnits)==len(keepProbs)
        assert min(keepProbs)>0 and max(keepProbs)<=1

        self.numCellUnits = numCellUnits
        self.keepProbs = keepProbs
        self.activation = activation


class Mark6(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 rnnConfigs,
                 pooledKeepProb, pooledActivation,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type rnnConfigs: list
        :type pooledKeepProb: float 
        """

        assert 0.0 < pooledKeepProb <= 1

        self.l2RegLambda = l2RegLambda
        self.pooledKeepProb = pooledKeepProb
        self.pooledActivation = pooledActivation
        self.rnnConfigs = rnnConfigs

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):
        makers = [RNNLayer.new(c.numCellUnits, c.keepProbs, activation=c.activation) for c in self.rnnConfigs]
        self.add_layers(makers, self.input, (-1, -1, self.vecDim))

        self.add_layers(DropoutLayer.new(self.pooledKeepProb, self.pooledActivation))
        lastLayer = self.add_layers(FullyConnectedLayer.new(self.numClasses))
        self.l2Loss = self.l2RegLambda * (tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases))

    @classmethod
    def quick_run(cls, runScale ='basic', dataScale='tiny_fake_2', useCPU = True):

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-3]),
                  ('rnnConfigs', [ [RNNConfig([16, 13], [1,1], 'relu')] ]),
                  # ('rnnCellUnitsNProbs', [ [([16, 13], [1, 1]), ([16, 13], [1, 1])] ]),
                  ('pooledKeepProb', [0.5]), ('pooledActivation', [None])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def comparison_run(cls, runScale='small', dataScale='full_2occupations', useCPU = True):
        params = [('initialLearningRate', [5e-4]),
                  ('l2RegLambda', [1e-4]),

                  ('rnnConfigs', [
                      [RNNConfig([128, 8], [0.8, 1]), RNNConfig([32, 8], [0.8, 1]), RNNConfig([16, 8], [0.8, 1])],
                      # [([128], [0.8]), ([32], [0.8]), ([16], [0.8])],
                      # [([128, 32, 16], [0.5, 0.8, 0.8]), ([64, 64], [0.5, 0.8]), ([32], [0.8])],
                  ]),

                  ('pooledKeepProb', [1]),
                  ('pooledActivation', ['relu'])
                  ]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def full_run(cls, runScale='tiny', dataScale='full', useCPU = True):
        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),

                  ('rnnConfigs', [
                      # [([128, 8], [0.8, 1]), ([32, 8], [0.8, 1]), ([16, 8], [0.8, 1])],
                      # [([16, 128], [1, 0.8]), ([8, 32], [1, 0.8]), ([8, 16], [1, 0.8])],
                      # [([128, 64, 16], [0.5, 0.5, 1]), ([16, 64, 128], [1, 0.5, 0.5]), ([32, 32, 32], [1]*3)],
                      [RNNConfig([128, 64, 16], [0.5]*3), RNNConfig([16, 64, 128], [0.5]*3), RNNConfig([32, 32, 32], [0.5]*3)],
                      [([64]*4, [0.5, 0.5, 0.5, 1]), ([32]*4, [0.5, 0.5, 0.5, 1]), ([16]*4,  [0.5, 0.5, 0.5, 1])],
                  ]),

                  ('pooledKeepProb', [1]),
                  ('pooledActivation', ['relu'])
                  ]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

if __name__ == '__main__':
    Mark6.full_run()
    # Mark6.comparison_run()
    # Mark6.quick_run()
    # Mark6.quick_learn()
