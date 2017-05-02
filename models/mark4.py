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
                 convFilterShapesNKeepProbs, convNumFeaturesPerFilter,
                 rnnCellUnitsNProbs,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type rnnCellUnitsNProbs: tuple
        """

        convFilterShapes, convKeepProbs = convFilterShapesNKeepProbs
        rnnNumCellUnits, rnnKeepProbs = rnnCellUnitsNProbs

        assert len(convFilterShapes) == len(convKeepProbs)
        assert len(rnnNumCellUnits) == len(rnnNumCellUnits)

        self.l2RegLambda = l2RegLambda
        self.maxNumSeqs = maxNumSeqs

        self.convFilterShapes = convFilterShapes
        self.convKeepProbs = convKeepProbs

        self.rnnNumCellUnits = rnnNumCellUnits
        self.rnnKeepProbs = rnnKeepProbs
        self.convNumFeaturesPerFilter = convNumFeaturesPerFilter

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):

        outputs = []
        outputShapes = []

        for filterShape, keepProb in zip(self.convFilterShapes, self.convKeepProbs):

            cnn = ConvLocalnormLayer(self.x, (-1, self.maxNumSeqs, self.vecDim),
                                     convParams_={'filterShape': (filterShape[0], self.vecDim if filterShape[1]==-1 else filterShape[1]),
                                                  'numFeaturesPerFilter': self.convNumFeaturesPerFilter,
                                                  'keepProb': keepProb,
                                                  'activation': 'relu'})

            newInput, newInputNumCols = convert_to_3d(cnn.output, cnn.output_shape)

            rnn = RNNLayer({'x': newInput, 'numSeqs': self.numSeqs - filterShape[0] + 1},
                           (-1, cnn.output_shape[1], newInputNumCols),
                           self.rnnNumCellUnits, self.rnnKeepProbs)

            outputs.append(rnn.output)
            outputShapes.append(rnn.output_shape)

        self.add_outputs(outputs, outputShapes)

        # last layer: fully connected
        lastLayer = self.add_layers(FullyConnectedLayer.new(self.numClasses))

        self.l2Loss = self.l2RegLambda * (tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases))


    @classmethod
    def quick_run(cls, runScale ='basic', dataScale='tiny_fake_2', useCPU = True):

        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('maxNumSeqs', [numSeqs]),
                  ('convFilterShapesNKeepProbs', [ ([(2, -1)],[1]) ]),
                  ('convNumFeaturesPerFilter', [6]),
                  ('rnnCellUnitsNProbs', [([3], [0.9])])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)

    @classmethod
    def quick_learn(cls, runScale ='small', dataScale='small_2occupations', useCPU = True):

        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-4]),
                  ('maxNumSeqs', [numSeqs]),
                  ('convFilterShapesNKeepProbs', [ ([(3, -1)], [1]) ]),
                  ('convNumFeaturesPerFilter', [32]),
                  ('rnnCellUnitsNProbs', [([16], [0.9])])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)


    @classmethod
    def comparison_run(cls, runScale ='small', dataScale='full_2occupations', useCPU = True):

        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-4]),
                  ('maxNumSeqs', [numSeqs]),
                  ('convFilterShapesNKeepProbs', [ ([(2, -1)],[0.8]),
                                                   ([(4, -1)], [0.8]),
                                                   ([(4, 4)], [0.8])]),
                  ('convNumFeaturesPerFilter', [64]),
                  ('rnnCellUnitsNProbs', [([32], [0.9])])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)

if __name__ == '__main__':
    # Mark4.quick_learn()
    # Mark4.quick_run()
    Mark4.comparison_run()
