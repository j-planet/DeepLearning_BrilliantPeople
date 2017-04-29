import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, DropoutWrapper

from utilities import last_relevant
from data_reader import DataReader
from layers.abstract_layer import AbstractLayer


# ------ stack of LSTM - bi-directional RNN layer ------
class RNNLayer(AbstractLayer):

    def __init__(self, input_, inputDim_,
                 numLSTMUnits_, outputKeepProbs_=1., numStepsToOutput_=1,
                 activation=None, loggerFactory=None):
        """
        :type input_: dict
        :type numLSTMUnits_: list
        :type numStepsToOutput_: int
        :type outputKeepProbs_: Union[list, float]
        """

        assert 'x' in input_ and 'numSeqs' in input_, 'Currently RNN works only as the top layer.'

        self.numLSTMUnits = numLSTMUnits_
        self.outputKeepProbs = [outputKeepProbs_] * len(numLSTMUnits_) if type(outputKeepProbs_) in [float, int] else outputKeepProbs_
        self.numStepsToOutput = numStepsToOutput_

        self.x = input_['x']
        self.numSeqs = input_['numSeqs']

        super().__init__(input_, inputDim_, activation, loggerFactory)

        self.print('# LSTM cell units: ' + str(numLSTMUnits_))
        self.print('dropout keep prob: %0.5f' % outputKeepProbs_)
        self.print('output %d steps' % numStepsToOutput_)

    def make_graph(self):

        self.forwardCells = self.make_stacked_cells()
        self.backwardCells = self.make_stacked_cells()

        self.outputs = tf.concat(
            tf.nn.bidirectional_dynamic_rnn(self.forwardCells, self.backwardCells,
                                            time_major=False, inputs=self.x, dtype=tf.float32,
                                            sequence_length=self.numSeqs,
                                            swap_memory=True)[0], 2)

        self.output = last_relevant(self.outputs, self.numSeqs, self.numStepsToOutput)

    def make_stacked_cells(self):

        # return MultiRNNCell([DropoutWrapper(BasicLSTMCell(f), output_keep_prob=k)
        #                      for f, k in zip(self.numLSTMUnits, self.outputKeepProbs)])
        return MultiRNNCell([BasicLSTMCell(f) for f in self.numLSTMUnits])

    @property
    def output_shape(self):
        return self.inputDim[0], self.numStepsToOutput, 2*self.numLSTMUnits[-1]

    @classmethod
    def new(cls, numLSTMUnits_, outputKeepProbs_=1., numStepsToOutput_=1,
            activation=None):

        return lambda input_, inputDim_, loggerFactory=None: \
            cls(input_, inputDim_, numLSTMUnits_, outputKeepProbs_, numStepsToOutput_, activation, loggerFactory)



if __name__ == '__main__':
    dr = DataReader('../data/peopleData/2_samples', 'bucketing', 10, 50)
    maker = RNNLayer.new([32, 16], [0.5, 1.], 3)
    layer = maker(dr.input, [10, -1])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print(sess.run(layer.output, dr.get_next_training_batch()[0]).shape)    # should be of shape 10 x 3 x 32