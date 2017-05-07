import tensorflow as tf

from models.abstract_model import AbstractModel
from data_readers.embedding_data_reader import EmbeddingDataReader
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.dropout_layer import DropoutLayer

from utilities import make_params_dict


class RNNConfig(object):
    def __init__(self, numCellUnits, keepProbs=1, activation=None):
        """
        :type keepProbs: Union[int, float, list] 
        """

        if type(keepProbs) in [float, int]:
            keepProbs = [keepProbs] * len(numCellUnits)

        assert len(numCellUnits)==len(keepProbs)
        assert min(keepProbs)>0 and max(keepProbs)<=1

        self.numCellUnits = numCellUnits
        self.keepProbs = keepProbs
        self.activation = activation


class Mark6(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 l2Scheme,
                 rnnConfigs,
                 pooledKeepProb, pooledActivation,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type l2Scheme: str
        :type rnnConfigs: list
        :type pooledKeepProb: float 
        """

        assert 0.0 < pooledKeepProb <= 1
        assert l2Scheme in ['final_stage', 'overall']

        self.l2RegLambda = l2RegLambda
        self.l2Scheme = l2Scheme
        self.pooledKeepProb = pooledKeepProb
        self.pooledActivation = pooledActivation
        self.rnnConfigs = rnnConfigs

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)
        self.print('l2 scheme: ' + l2Scheme)

    def make_graph(self):
        makers = [RNNLayer.new(c.numCellUnits, c.keepProbs, activation=c.activation) for c in self.rnnConfigs]
        self.add_layers(makers, self.input, (-1, -1, self.vecDim))

        self.add_layers(DropoutLayer.new(self.pooledKeepProb, self.pooledActivation))
        lastLayer = self.add_layers(FullyConnectedLayer.new(self.numClasses))

        self.l2Loss = self.l2RegLambda * (
            tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases) if self.l2Scheme=='final_stage'
            else sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        )

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
    def full_run(cls, runScale='full', dataScale='full', useCPU = True):

        def _p(start, count, pattern):
            """
            produce a list of dropout probs
            """

            assert pattern in ['inc', 'dec', 'constant']

            if pattern=='constant':
                return [start]*count

            delta = 0.1 if pattern=='inc' else -0.1

            res = [start]

            for _ in range(count-1):
                res.append( max(min(res[-1] + delta, 1), 0.1) )

            return res

        def _c(start, count, pattern):
            """
            produce a list of number of cell units, by halfing or doubling
            """

            assert pattern in ['inc', 'dec', 'constant']

            if pattern == 'constant':
                return [start] * count


            delta = 2 if pattern == 'inc' else 0.5

            res = [start]

            for _ in range(count - 1):
                res.append(max(min(res[-1]*delta, 2048), 8))

            return res

        rnnConfigs = []

        for pd in [0.6, 0.7, 1]:
            for pd_pattern in ['inc', 'dec', 'constant']:
                for numLayers in [2,3,4,5]:
                    for c in [1024, 256, 64, 32, 16]:
                        for c_pattern in ['inc', 'dec', 'constant']:

                            rnnConfigs.append([RNNConfig(_c(c, numLayers, c_pattern), _p(pd, numLayers, pd_pattern))])


        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-6]),
                  ('l2Scheme', ['overall', 'final_stage']),

                  ('rnnConfigs', rnnConfigs),

                  ('pooledKeepProb', [0.5, 0.9, 1]),
                  ('pooledActivation', [None, 'relu'])
                  ]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

if __name__ == '__main__':
    Mark6.full_run()
    # Mark6.full_run('small')
    # Mark6.comparison_run()
    # Mark6.quick_run()
    # Mark6.quick_learn()
