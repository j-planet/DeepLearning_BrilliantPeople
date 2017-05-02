# Mark 3: Embeddings followed by CNN-maxpool
# with conv done sequence-wise and maxpool done embedding dimension-wise
# just as in http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

import tensorflow as tf
import numpy as np
import os

from models.abstract_model import AbstractModel
from data_readers.text_data_reader import TextDataReader
from layers.embedding_layer import EmbeddingLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.dropout_layer import DropoutLayer
from layers.conv_maxpool_layer import ConvMaxpoolLayer

from train import train
from utilities import run_with_processor, make_params_dict

PPL_DATA_DIR = '../data/peopleData/'




class Mark3d(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 vocabSize, embeddingDim,
                 convFilterSizesNKeepProbs, numFeaturesPerFilter,
                 pooledKeepProb,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type pooledKeepProb: float 
        :type vocabSize: int
        :type embeddingDim: int
        :type convFilterSizesNKeepProbs: tuple
        """

        filterSizes, convKeepProbs = convFilterSizesNKeepProbs

        assert len(filterSizes) == len(convKeepProbs)

        self.l2RegLambda = l2RegLambda
        self.pooledKeepProb = pooledKeepProb
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.filterSizes = filterSizes
        self.convKeepProbs = convKeepProbs
        self.numFeaturesPerFilter = numFeaturesPerFilter

        super().__init__(input_, initialLearningRate, loggerFactory_)
        self.print('l2 reg lambda: %0.7f' % l2RegLambda)

    def make_graph(self):

        inputNumCols = self.input['x'].get_shape()[1].value

        # layer1: embedding
        layer1 = self.add_layer(EmbeddingLayer.new(self.vocabSize, self.embeddingDim),
                                self.input['x'], (-1, inputNumCols))

        # layer2: a bunch of conv-maxpools
        layer2_outputs = []

        for filterSize, keepProb in zip(self.filterSizes, self.convKeepProbs):

            l = ConvMaxpoolLayer(layer1.output, layer1.output_shape,
                                 convParams_={'filterShape': (filterSize, self.embeddingDim),
                                              'numFeaturesPerFilter': self.numFeaturesPerFilter,
                                              'keepProb': keepProb,
                                              'activation': 'relu'},
                                 maxPoolParams_={'ksize': (inputNumCols - filterSize + 1, 1), 'padding': 'VALID'},
                                 loggerFactory=self.loggerFactory)

            layer2_outputs.append(l.output)


        layer2_outputShape = -1, self.numFeaturesPerFilter * len(self.filterSizes)
        layer2_output = tf.reshape(tf.concat(layer2_outputs, 3), layer2_outputShape)

        self.add_output(layer2_output, layer2_outputShape)

        # layer3: dropout
        self.add_layer(DropoutLayer.new(self.pooledKeepProb))

        # layer4: fully connected
        lastLayer = self.add_layer(FullyConnectedLayer.new(self.numClasses, activation='relu'))

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
                  ('convFilterSizesNKeepProbs', [([2,4], [0.9, 0.9])]),
                  ('numFeaturesPerFilter', [8]),
                  ('pooledKeepProb', [1])]

        cls.run_thru_data(TextDataReader, dataScale, make_params_dict(params), runScale, useCPU)

    @classmethod
    def quick_learn(cls, runScale ='small', dataScale='full_2occupations', useCPU = True):

        # ok this is silly. But at least it's fast.
        vocabSize = TextDataReader.maker_from_premade_source(dataScale)(
            bucketingOrRandom = 'bucketing', batchSize_ = 50, minimumWords = 0).vocabSize

        params = [('initialLearningRate', [1e-4]),
                  ('l2RegLambda', [1e-5]),
                  ('vocabSize', [vocabSize]),
                  ('embeddingDim', [256]),
                  ('convFilterSizesNKeepProbs', [([2, 3, 5], [0.6, 0.6, 0.6])]),
                  ('numFeaturesPerFilter', [32]),
                  ('pooledKeepProb', [0.9])]

        cls.run_thru_data(TextDataReader, dataScale, make_params_dict(params), runScale, useCPU)


    @classmethod
    def full_run(cls, runScale='full', dataScale='full', useCPU=True):
        # ok this is silly. But at least it's fast.
        vocabSize = TextDataReader.maker_from_premade_source(dataScale)(
            bucketingOrRandom='bucketing', batchSize_=50, minimumWords=0).vocabSize

        params = [('initialLearningRate', [1e-3]),
                  ('l2RegLambda', [0, 1e-4, 1e-5]),
                  ('vocabSize', [vocabSize]),
                  ('embeddingDim', [64, 128, 300]),
                  ('convFilterSizesNKeepProbs', [([1, 2, 3], [0.6, 0.6, 0.6]),
                                                 ([1, 2, 3], [0.7, 0.7, 0.9]),
                                                 ([2, 3, 4], [0.6, 0.6, 0.6]),
                                                 ([2, 3, 4], [0.7, 0.7, 0.9]),
                                                 ([3, 5, 10, 15], [0.7]*4),
                                                 ]),
                  ('numFeaturesPerFilter', [16, 32, 64]),
                  ('pooledKeepProb', [0.5, 0.7, 0.9, 1])]

        cls.run_thru_data(TextDataReader, dataScale, make_params_dict(params), runScale, useCPU)

if __name__ == '__main__':
    Mark3d.quick_learn()