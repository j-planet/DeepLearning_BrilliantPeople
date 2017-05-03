import tensorflow as tf

from models.abstract_model import AbstractModel
from data_readers.embedding_data_reader import EmbeddingDataReader
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.dropout_layer import DropoutLayer
from layers.conv_localnorm_layer import ConvLocalnormLayer

from utilities import make_params_dict, convert_to_2d



class Mark5(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 maxNumSeqs,
                 rnnCellUnitsNProbs,
                 convFilterSizesNKeepProbs, convNumFeaturesPerFilter,
                 pooledKeepProb,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type rnnCellUnitsNProbs: tuple
        :type pooledKeepProb: float 
        """

        rnnNumCellUnits, rnnKeepProbs = rnnCellUnitsNProbs
        convFilterSizes, convKeepProbs = convFilterSizesNKeepProbs

        assert len(rnnNumCellUnits) == len(rnnNumCellUnits)
        assert len(convFilterSizes) == len(convKeepProbs)
        assert 0.0 < pooledKeepProb <= 1

        self.l2RegLambda = l2RegLambda
        self.maxNumSeqs = maxNumSeqs

        self.rnnNumCellUnits = rnnNumCellUnits
        self.rnnKeepProbs = rnnKeepProbs

        self.convFilterSizes = convFilterSizes
        self.convKeepProbs = convKeepProbs
        self.convNumFeaturesPerFilter = convNumFeaturesPerFilter

        self.pooledKeepProb = pooledKeepProb

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):

        # -------------- RNN --------------
        rnn = RNNLayer(self.input, (-1, -1, self.vecDim),
                       self.rnnNumCellUnits, self.rnnKeepProbs,
                       loggerFactory=self.loggerFactory)

        rnn_output, rnn_numcols = convert_to_2d(rnn.output, rnn.output_shape)

        # -------------- CNN --------------
        cnn_outputs = []
        cnn_numcols = 0
        cnn_layers = []

        for filterSize, keepProb in zip(self.convFilterSizes, self.convKeepProbs):

            l = ConvLocalnormLayer(self.x, (-1, self.maxNumSeqs, self.vecDim),
                                   convParams_={'filterShape': (filterSize, self.vecDim),
                                                'numFeaturesPerFilter': self.convNumFeaturesPerFilter,
                                                'keepProb': keepProb,
                                                'activation': 'relu'},
                                   loggerFactory=self.loggerFactory)

            o, col = convert_to_2d(l.output, l.output_shape)
            cnn_outputs.append(o)
            cnn_numcols += col
            cnn_layers.append(l)

        cnn_output = tf.concat(cnn_outputs, axis=1)

        # -------------- combine RNN & CNN --------------
        self.layers.append([rnn, cnn_layers])
        self.add_outputs([rnn_output, cnn_output], [(-1, rnn_numcols), (-1, cnn_numcols)])

        # -------------- dropout --------------
        self.add_layers(DropoutLayer.new(self.pooledKeepProb))

        # -------------- fully connected --------------
        lastLayer = self.add_layers(FullyConnectedLayer.new(self.numClasses))

        self.l2Loss = self.l2RegLambda * (tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases))


    @classmethod
    def quick_run(cls, runScale ='basic', dataScale='tiny_fake_2', useCPU = True):

        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('maxNumSeqs', [numSeqs]),

                  ('rnnCellUnitsNProbs', [([3], [0.9])
                                          ]),

                  ('convFilterSizesNKeepProbs', [([2], [1.])
                                                 ]),
                  ('convNumFeaturesPerFilter', [4]),

                  ('pooledKeepProb', [1])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)

    @classmethod
    def comparison_run(cls, runScale='full', dataScale='full_2occupations', useCPU = True):

        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-6]),
                  ('maxNumSeqs', [numSeqs]),

                  ('rnnCellUnitsNProbs', [([128, 128, 64], [1, 1, 1]),
                                          ([128, 128, 64], [0.8, 0.8, 0.9])]),

                  ('convFilterSizesNKeepProbs', [([1, 2, 4], [1, 1, 1]),
                                                 ([1, 2, 4], [0.8, 0.8, 0.9])]),
                  ('convNumFeaturesPerFilter', [32]),

                  ('pooledKeepProb', [0.8, 0.6])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)

    @classmethod
    def full_run(cls, runScale='full', dataScale='full', useCPU = True):

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-5]),
                  ('numRnnOutputSteps', [10]),
                  ('rnnCellUnitsNProbs', [([128, 64, 64], [0.8, 0.8, 0.9]),
                                          ]),
                  ('convNumFeaturesPerFilter', [16]),
                  ('pooledKeepProb', [0.5, 0.9])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

if __name__ == '__main__':
    Mark5.comparison_run()
    # Mark5.quick_run()
