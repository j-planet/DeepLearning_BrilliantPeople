import tensorflow as tf
import numpy as np

from models.abstract_model import AbstractModel
from data_reader import DataReader
from layers.rnn_layer import RNNLayer
from layers.conv_layer import ConvLayer
from layers.maxpool_layer import MaxpoolLayer
from layers.fully_connected_layer import FullyConnectedLayer



class Model(AbstractModel):
    def __init__(self, input_, loggerFactory_=None):
        self.l2RegLambda = 0
        self.initialLearningRate = 1e-3
        super().__init__(input_, loggerFactory_)


    def make_graph(self):
        rnnOutputSteps = 3

        layer1 = RNNLayer(self.input, [16], numStepsToOutput_ = rnnOutputSteps)   # bs x numSeq x (2*last num hidden units)
        prevOutputShape = layer1.output_size(-1)
        prevOutput = layer1.output
        self.outputs.append({'output': prevOutput, 'size': prevOutputShape})

        layer2 = ConvLayer(prevOutput, filterShape=(1, rnnOutputSteps), numFeaturesPerFilter=8, activation='relu')
        prevOutputShape = layer2.output_size(prevOutputShape)
        prevOutput = layer2.output
        self.outputs.append({'output': prevOutput, 'size': prevOutputShape})

        layer3 = MaxpoolLayer(prevOutput, ksize=(1, 1))
        prevOutputShape = layer3.output_size(prevOutputShape)
        prevOutput = layer3.output
        self.outputs.append({'output': prevOutput, 'size': prevOutputShape})

        curInput = tf.reshape(prevOutput, [-1, np.product(prevOutputShape[1:])])
        layer4 = FullyConnectedLayer(curInput, (curInput.get_shape()[-1].value, self.numClasses), None)
        prevOutputShape = layer4.output_size(-1)
        prevOutput = layer4.output
        self.outputs.append({'output': prevOutput, 'size': prevOutputShape})

        self.output = prevOutput


if __name__ == '__main__':

    datadir = '../data/peopleData/2_samples'
    # datadir = '../data/peopleData/earlyLifesWordMats/politician_scientist'

    batchSize = 20
    dr = DataReader(datadir, 'bucketing', 20)
    model = Model(dr.input)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(24):
        fd = dr.get_next_training_batch()[0]
        _, c, acc = model.train_op(sess, fd, True)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))