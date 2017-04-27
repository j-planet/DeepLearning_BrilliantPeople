import tensorflow as tf

from models.abstract_model import AbstractModel
from data_reader import DataReader
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer



class Model(AbstractModel):
    def __init__(self, input_, initialLearningRate_, l2RegLambda_, loggerFactory_=None):
        super().__init__(input_, initialLearningRate_, l2RegLambda_, loggerFactory_)

    def make_graph(self):
        prevOutput = RNNLayer(self.input, [2, 5], [0.5, 0.8], 1, self.loggerFactory).output   # bs x numSeq x vecDim

        curInput = tf.reshape(prevOutput, [-1, prevOutput.get_shape()[-1].value])   # make 3D into 2D
        return FullyConnectedLayer(curInput, self.numClasses, None, self.loggerFactory).output

if __name__ == '__main__':

    dr = DataReader('../data/peopleData/2_samples', 'bucketing', 20)
    model = Model(dr.input, 1e-3, 1e-3)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(10):
        _, c, acc = model.train_op(sess, dr.get_next_training_batch()[0], True)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))