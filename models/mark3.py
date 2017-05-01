import tensorflow as tf
import numpy as np

from models.abstract_model import AbstractModel
from data_readers import DataReader_Text
from layers.embedding_layer import EmbeddingLayer
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.dropout_layer import DropoutLayer
from layers.conv_maxpool_layer import ConvMaxpoolLayer


def convert_to_2d(t, d):
    newSecondD = np.product(d[1:])
    return tf.reshape(t, [-1, newSecondD]), newSecondD

class Mark3(AbstractModel):

    def __init__(self, input_,
                 initialLearningRate, l2RegLambda,
                 vocabSize, embeddingDim,
                 filterSizes, numFeaturesPerFilter,
                 pooledKeepProb,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type l2RegLambda: float
        :type pooledKeepProb: float 
        :type vocabSize: int
        :type embeddingDim: int
        """


        self.l2RegLambda = l2RegLambda
        self.pooledKeepProb = pooledKeepProb
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.filterSizes = filterSizes
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

        for filterSize in self.filterSizes:

            l = ConvMaxpoolLayer(layer1.output, layer1.output_shape,
                                 convParams_={'filterShape': (filterSize, self.embeddingDim),
                                              'numFeaturesPerFilter': self.numFeaturesPerFilter, 'activation': 'relu'},
                                 maxPoolParams_={'ksize': (inputNumCols - filterSize + 1, 1), 'padding': 'VALID'},
                                 loggerFactory=self.loggerFactory)

            layer2_outputs.append(l.output)


        layer2_outputShape = -1, self.numFeaturesPerFilter * len(self.filterSizes)
        layer2_output = tf.reshape(tf.concat(layer2_outputs, 3), layer2_outputShape)

        self.add_output(layer2_output, layer2_outputShape)

        # layer3: dropout
        self.add_layer(DropoutLayer.new(self.pooledKeepProb))

        # layer4: fully connected
        lastLayer = self.add_layer(FullyConnectedLayer.new(self.numClasses))

        self.l2Loss = self.l2RegLambda * (tf.nn.l2_loss(lastLayer.weights) + tf.nn.l2_loss(lastLayer.biases))



if __name__ == '__main__':


    lr = 1e-3
    # dr = DataReader_Text('../data/peopleData/tokensfiles/pol_sci.json', 'bucketing', 40, 30)
    dr = DataReader_Text('../data/peopleData/tokensfiles/fake2.json', 'bucketing', 40, 30)

    model = Mark3(dr.input, lr, 0,
                  dr.vocabSize, embeddingDim=16,
                  filterSizes=[2,3,5], numFeaturesPerFilter=4,
                  pooledKeepProb=1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        if step % 10 == 0:
            print('Lowering learning rate to', lr)
            lr *= 0.9
            model.assign_lr(sess, lr)

        fd = dr.get_next_training_batch()[0]
        _, c, acc = model.train_op(sess, fd, True)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))