# Mark 3: Embeddings followed by CNN-maxpool
# with conv done sequence-wise and maxpool done embedding dimension-wise
# just as in http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

import tensorflow as tf

from models.abstract_model import AbstractModel
from data_readers.embedding_data_reader import EmbeddingDataReader
from layers.fully_connected_layer import FullyConnectedLayer
from layers.dropout_layer import DropoutLayer
from layers.conv_maxpool_layer import ConvMaxpoolLayer
from layers.conv_localnorm_layer import ConvLocalnormLayer

from utilities import make_params_dict, convert_to_2d

PPL_DATA_DIR = '../data/peopleData/'


class Mark3e(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 maxNumSeqs,
                 filterSizesNKeepProbs, numFeaturesPerFilter,
                 pooledKeepProb,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type pooledKeepProb: float 
        """

        filterSizes, keepProbs = filterSizesNKeepProbs
        assert len(filterSizes) == len(keepProbs)

        self.l2RegLambda = l2RegLambda
        self.pooledKeepProb = pooledKeepProb
        self.maxNumSeqs = maxNumSeqs
        self.filterSizes = filterSizes
        self.keepProbs = keepProbs
        self.numFeaturesPerFilter = numFeaturesPerFilter

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):

        # layer1: a bunch of conv-maxpools
        layer1_outputs = []
        layer1_numcols = 0
        layer1_layers = []

        for filterSize, keepProb in zip(self.filterSizes, self.keepProbs):

            l = ConvLocalnormLayer(self.x, (-1, self.maxNumSeqs, self.vecDim),
                                 convParams_={'filterShape': (filterSize, self.vecDim),
                                              'numFeaturesPerFilter': self.numFeaturesPerFilter,
                                              'keepProb': keepProb,
                                              'activation': 'relu'},
                                 loggerFactory=self.loggerFactory)

            o, col = convert_to_2d(l.output, l.output_shape)
            layer1_outputs.append(o)
            layer1_numcols += col
            layer1_layers.append(l)

        self.layers.append(layer1_layers)

        layer1_output = tf.concat(layer1_outputs, axis=1)

        self.add_output(layer1_output, (-1, layer1_numcols))

        # layer2: dropout
        self.add_layers(DropoutLayer.new(self.pooledKeepProb))

        # layer3: fully connected
        lastLayer = self.add_layers(FullyConnectedLayer.new(self.numClasses))

        self.l2Loss = self.l2RegLambda * (tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases))


    @classmethod
    def quick_run(cls, runScale ='tiny', dataScale='tiny_fake_2', useCPU = True):

        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('maxNumSeqs', [numSeqs]),
                  ('filterSizesNKeepProbs', [([2, 4], [1, 1])]),
                  ('numFeaturesPerFilter', [3]),
                  ('pooledKeepProb', [1])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)

    @classmethod
    def quick_learn(cls, runScale ='small', dataScale='small_2occupations', useCPU = True):
        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('maxNumSeqs', [numSeqs]),
                  ('filterSizes', [[2, 4]]),
                  ('numFeaturesPerFilter', [3]),
                  ('pooledKeepProb', [1])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)


    @classmethod
    def comparison_run(cls, runScale='medium', dataScale='full_2occupations', useCPU=True):
        numSeqs = EmbeddingDataReader(EmbeddingDataReader.premade_sources()[dataScale], 'bucketing', 100, 40, padToFull=True).maxXLen

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-6]),
                  ('maxNumSeqs', [numSeqs]),
                  ('filterSizesNKeepProbs', [([1, 2, 3, 4], [0.9, 0.9, 0.9, 0.9])]),
                  ('numFeaturesPerFilter', [128]),
                  ('pooledKeepProb', [0.5, 0.85, 1])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)

    @classmethod
    def full_run(cls, runScale='full', dataScale='full', useCPU=True):

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0, 1e-5]),
                  ('filterSizes', [[1, 2, 4], [3, 5, 10, 15]]),
                  ('numFeaturesPerFilter', [16, 32, 64]),
                  ('pooledKeepProb', [0.5, 0.7, .9, 1])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU, padToFull=True)

if __name__ == '__main__':
    # Mark3b.quick_learn('tiny', 'tiny_fake_2')
    # Mark3e.quick_run()
    Mark3e.comparison_run(runScale='full')
