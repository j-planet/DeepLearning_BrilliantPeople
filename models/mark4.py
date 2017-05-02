import tensorflow as tf

from models.abstract_model import AbstractModel
from data_readers.embedding_data_reader import EmbeddingDataReader
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.conv_localnorm_layer import ConvLocalnormLayer

from utilities import make_params_dict, convert_to_3d



class Mark4(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 maxNumSeqs,
                 convFilterSizesNKeepProbs, convNumFeaturesPerFilter,
                 rnnCellUnitsNProbs,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type rnnCellUnitsNProbs: tuple
        """

        convFilterSizes, convKeepProbs = convFilterSizesNKeepProbs
        rnnNumCellUnits, rnnKeepProbs = rnnCellUnitsNProbs

        assert len(convFilterSizes) == len(convKeepProbs)
        assert len(rnnNumCellUnits) == len(rnnNumCellUnits)

        self.l2RegLambda = l2RegLambda
        self.maxNumSeqs = maxNumSeqs

        self.convFilterSizes = convFilterSizes
        self.convKeepProbs = convKeepProbs

        self.rnnNumCellUnits = rnnNumCellUnits
        self.rnnKeepProbs = rnnKeepProbs
        self.convNumFeaturesPerFilter = convNumFeaturesPerFilter

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):

        # layer 1's: CNN
        self.add_layers(
            [ConvLocalnormLayer.new(convParams_={'filterShape': (filterSize, self.vecDim),
                                                 'numFeaturesPerFilter': self.convNumFeaturesPerFilter,
                                                 'keepProb': keepProb,
                                                 'activation': 'relu'})
             for (filterSize, keepProb) in zip(self.convFilterSizes, self.convKeepProbs)],
            self.x,
            (-1, self.maxNumSeqs, self.vecDim)
        )

        # layer 2: RNN
        newInput, newInputNumCols = convert_to_3d(self.prevOutput, self.prevOutputShape)
        self.add_layers(RNNLayer.new(self.rnnNumCellUnits, self.rnnKeepProbs),
                        input_={'x': newInput, 'numSeqs': self.numSeqs - self.convFilterSizes[0]+1},    # TODO: works only for 1 filter for now
                        inputDim_=(-1, self.prevOutputShape[1], newInputNumCols))

        # last layer: fully connected
        lastLayer = self.add_layers(FullyConnectedLayer.new(self.numClasses))

        self.l2Loss = self.l2RegLambda * (tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases))


    @classmethod
    def quick_run(cls, runScale ='basic', dataScale='tiny_fake_2', useCPU = True):

        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('maxNumSeqs', [numSeqs]),
                  ('convFilterSizesNKeepProbs', [ ([2], [1]) ]),
                  ('convNumFeaturesPerFilter', [6]),
                  ('rnnCellUnitsNProbs', [([3], [0.9])])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)

    @classmethod
    def quick_learn(cls, runScale ='small', dataScale='small_2occupations', useCPU = True):

        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-4]),
                  ('maxNumSeqs', [numSeqs]),
                  ('convFilterSizesNKeepProbs', [ ([3], [1]) ]),
                  ('convNumFeaturesPerFilter', [32]),
                  ('rnnCellUnitsNProbs', [([16], [0.9])])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)


if __name__ == '__main__':
    Mark4.quick_learn()
    # Mark2d.quick_run()
