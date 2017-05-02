import tensorflow as tf

from models.abstract_model import AbstractModel
from data_readers.embedding_data_reader import EmbeddingDataReader
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.dropout_layer import DropoutLayer
from layers.conv_maxpool_layer import ConvMaxpoolLayer

from utilities import make_params_dict, convert_to_2d



class Mark2b(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
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
        self.numRnnOutputSteps = numRnnOutputSteps
        self.rnnNumCellUnits = rnnNumCellUnits
        self.rnnKeepProbs = rnnKeepProbs
        self.convNumFeaturesPerFilter = convNumFeaturesPerFilter
        self.pooledKeepProb = pooledKeepProb

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):

        layer1 = self.add_layer(RNNLayer.new(
            self.rnnNumCellUnits, numStepsToOutput_ = self.numRnnOutputSteps), self.input, (-1, -1, self.vecDim))

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

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('numRnnOutputSteps', [10]),
                  ('rnnCellUnitsNProbs', [([3], [1]),
                                          ([4, 8], [1, 1])]),
                  ('convNumFeaturesPerFilter', [16]),
                  ('pooledKeepProb', [1])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def quick_learn(cls, runScale ='small', dataScale='small_2occupations', useCPU = True):

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0]),
                  ('numRnnOutputSteps', [10]),
                  ('rnnCellUnitsNProbs', [([32], [0.7]),
                                          ([64, 16], [0.6, 0.9])]),
                  ('convNumFeaturesPerFilter', [16]),
                  ('pooledKeepProb', [1])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def full_run(cls, runScale='full', dataScale='full', useCPU = True):

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [1e-4, 1e-5]),
                  ('numRnnOutputSteps', [5, 10, 40]),
                  ('rnnCellUnitsNProbs', [([128, 64], [0.5, 0.9]),
                                          ([64, 64, 32], [0.8, 0.8, 0.9]),
                                          ([128, 128, 64, 64], [0.5, 0.7, 0.8, 0.9])]),
                  ('convNumFeaturesPerFilter', [16, 32]),
                  ('pooledKeepProb', [0.5, 0.9])]

        cls.run_thru_data(EmbeddingDataReader, dataScale, make_params_dict(params), runScale, useCPU)

if __name__ == '__main__':
    # Mark2b.quick_run()
    Mark2b.quick_learn()
