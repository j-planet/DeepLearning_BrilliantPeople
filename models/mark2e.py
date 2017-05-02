import tensorflow as tf

from data_readers.text_data_reader import TextDataReader
from layers.conv_maxpool_layer import ConvMaxpoolLayer
from layers.dropout_layer import DropoutLayer
from layers.embedding_layer import EmbeddingLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.rnn_layer import RNNLayer
from models.abstract_model import AbstractModel
from utilities import make_params_dict, convert_to_2d


class Mark2e(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 vocabSize, embeddingDim,
                 numRnnOutputSteps, rnnCellUnitsNProbs,
                 convNumFeaturesPerFilter,
                 pooledKeepProb,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type rnnCellUnitsNProbs: tuple
        :type numRnnOutputSteps: int 
        :type pooledKeepProb: float 
        """

        rnnNumCellUnits, rnnKeepProbs = rnnCellUnitsNProbs
        assert len(rnnNumCellUnits) == len(rnnNumCellUnits)
        assert 0.0 < pooledKeepProb <= 1

        self.l2RegLambda = l2RegLambda
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.numRnnOutputSteps = numRnnOutputSteps
        self.rnnNumCellUnits = rnnNumCellUnits
        self.rnnKeepProbs = rnnKeepProbs
        self.convNumFeaturesPerFilter = convNumFeaturesPerFilter
        self.pooledKeepProb = pooledKeepProb

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):

        inputNumCols = self.x.get_shape()[1].value

        # layer0: embedding
        layer0 = self.add_layer(EmbeddingLayer.new(self.vocabSize, self.embeddingDim),
                       self.x, (-1, inputNumCols))

        layer1 = self.add_layer(RNNLayer.new(self.rnnNumCellUnits,
                                             self.rnnKeepProbs,
                                             numStepsToOutput_ = self.numRnnOutputSteps),
                                input_={'x': layer0.output, 'numSeqs': self.numSeqs},
                                inputDim_=(-1, self.vocabSize, self.embeddingDim))

        # just last row of the rnn output
        numCols = layer1.output_shape[2]

        layer2a_output = layer1.output[:,-1,:]
        layer2a_outputshape = (layer1.output_shape[0], numCols)

        layer2b = ConvMaxpoolLayer(layer1.output, layer1.output_shape,
                                   convParams_={'filterShape': (2, numCols),
                                                'numFeaturesPerFilter': self.convNumFeaturesPerFilter,
                                                'activation': 'relu'},
                                   maxPoolParams_={'ksize': (self.numRnnOutputSteps, 1), 'padding': 'SAME'},
                                   loggerFactory=self.loggerFactory)
        layer2b_output, layer2b_output_numcols = convert_to_2d(layer2b.output, layer2b.output_shape)


        layer2c = ConvMaxpoolLayer(layer1.output, layer1.output_shape,
                                   convParams_={'filterShape': (4, numCols),
                                                'numFeaturesPerFilter': self.convNumFeaturesPerFilter,
                                                'activation': 'relu'},
                                   maxPoolParams_={'ksize': (self.numRnnOutputSteps, 1), 'padding': 'SAME'},
                                   loggerFactory=self.loggerFactory)
        layer2c_output, layer2c_output_numcols = convert_to_2d(layer2c.output, layer2c.output_shape)

        layer2_output = tf.concat([layer2a_output, layer2b_output, layer2c_output], axis=1)
        layer2_output_numcols = layer2a_outputshape[1] + layer2b_output_numcols + layer2c_output_numcols

        self.layers.append([layer2b, layer2c])
        self.outputs.append({'output': layer2_output,
                             'output_shape': (layer2a_outputshape[0], layer2_output_numcols)})

        self.add_layer(DropoutLayer.new(self.pooledKeepProb))

        lastLayer = self.add_layer(FullyConnectedLayer.new(self.numClasses))

        self.l2Loss = self.l2RegLambda * (tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases))


    @classmethod
    def quick_run(cls, runScale ='basic', dataScale='tiny_fake_2', useCPU = True):

        # ok this is silly. But at least it's fast.
        vocabSize = TextDataReader.maker_from_premade_source(dataScale)(
            bucketingOrRandom = 'bucketing', batchSize_ = 50, minimumWords = 0).vocabSize

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('vocabSize', [vocabSize]),
                  ('embeddingDim', [32]),
                  ('numRnnOutputSteps', [10]),
                  ('rnnCellUnitsNProbs', [([3], [0.9]),
                                          ([4, 8], [1, 1])]),
                  ('convNumFeaturesPerFilter', [16]),
                  ('pooledKeepProb', [1])]

        cls.run_thru_data(TextDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def quick_learn(cls, runScale ='small', dataScale='full_2occupations', useCPU = True):

        # ok this is silly. But at least it's fast.
        vocabSize = TextDataReader.maker_from_premade_source(dataScale)(
            bucketingOrRandom = 'bucketing', batchSize_ = 50, minimumWords = 0).vocabSize

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('vocabSize', [vocabSize]),
                  ('embeddingDim', [32]),
                  ('numRnnOutputSteps', [10]),
                  ('rnnCellUnitsNProbs', [([8, 4], [0.5, 0.9])]),
                  ('convNumFeaturesPerFilter', [16]),
                  ('pooledKeepProb', [1])]

        cls.run_thru_data(TextDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def comparison_run(cls, runScale ='medium', dataScale='full_2occupations', useCPU = True):

        # ok this is silly. But at least it's fast.
        vocabSize = TextDataReader.maker_from_premade_source(dataScale)(
            bucketingOrRandom = 'bucketing', batchSize_ = 50, minimumWords = 0).vocabSize

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [5e-4]),
                  ('vocabSize', [vocabSize]),
                  ('embeddingDim', [128, 300]),
                  ('numRnnOutputSteps', [5, 10]),
                  ('rnnCellUnitsNProbs', [([64, 64, 32], [0.8, 0.8, 0.9])]),
                  ('convNumFeaturesPerFilter', [16]),
                  ('pooledKeepProb', [0.5, 0.9])]

        cls.run_thru_data(TextDataReader, dataScale, make_params_dict(params), runScale, useCPU)

if __name__ == '__main__':
    Mark2e.quick_learn()
