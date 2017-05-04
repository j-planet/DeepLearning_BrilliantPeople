import tensorflow as tf

from models.abstract_model import AbstractModel
from data_readers.embedding_data_reader import EmbeddingDataReader
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.dropout_layer import DropoutLayer

from utilities import make_params_dict



class Mark6(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 rnnCellUnitsNProbs,
                 pooledKeepProb,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type rnnCellUnitsNProbs: list
        :type pooledKeepProb: float 
        """

        assert 0.0 < pooledKeepProb <= 1

        self.l2RegLambda = l2RegLambda
        self.pooledKeepProb = pooledKeepProb
        self.rnnCellUnitsNProbs = rnnCellUnitsNProbs

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):
        makers = [RNNLayer.new(cellUnits, keepProbs) for cellUnits, keepProbs in self.rnnCellUnitsNProbs]
        self.add_layers(makers, self.input, (-1, -1, self.vecDim))

        self.add_layers(DropoutLayer.new(self.pooledKeepProb))
        lastLayer = self.add_layers(FullyConnectedLayer.new(self.numClasses))
        self.l2Loss = self.l2RegLambda * (tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases))

    @classmethod
    def quick_run(cls, runScale ='basic', dataScale='tiny_fake_2', useCPU = True):

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-3]),
                  ('rnnCellUnitsNProbs', [ [([16, 13], [1, 1]), ([16, 13], [1, 1])] ]),
                  ('pooledKeepProb', [0.5])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def quick_learn(cls, runScale ='small', dataScale='full_2occupations', useCPU = True):

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-3]),
                  ('rnnCellUnitsNProbs', [[([32, 16], [1, 1]), ([8], [1])]]),
                  ('pooledKeepProb', [0.5])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def comparison_run(cls, runScale='full', dataScale='full_2occupations', useCPU = True):
        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-4]),

                  ('rnnCellUnitsNProbs', [
                      [([128], [1]), ([32], [1]), ([16], [1])],
                      [([128, 32, 16], [1, 1, 1]), ([64, 64], [1, 1])],
                      [([128, 32, 16], [0.5, 0.8, 0.8]), ([64, 64], [0.5, 0.8])],
                                           ]),

                  ('pooledKeepProb', [0.5])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)


if __name__ == '__main__':
    Mark6.comparison_run()
    # Mark6.quick_run()
    # Mark6.quick_learn()
