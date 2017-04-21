import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow import summary
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, DropoutWrapper

from data_reader import DataReader


def last_relevant(output_, lengths_):
    batch_size = tf.shape(output_)[0]
    max_length = tf.shape(output_)[1]
    out_size = int(output_.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (lengths_ - 1)
    flat = tf.reshape(output_, [-1, out_size])

    return tf.gather(flat, index)


class Model2Config(object):
    def __init__(self, scale, loggerFactory=None):
        assert scale in ['basic', 'tiny', 'small', 'full']

        if scale == 'basic':
            self.initialLearningRate = 0.002
            self.layer1CellUnits = [8]
            self.layer1keepProbs = [0.75]
            self.layer2CellUnits = 4
            self.layer2keepProb = 0.9

        elif scale == 'tiny':
            self.initialLearningRate = 0.002
            self.layer1CellUnits = [32, 8]
            self.layer1keepProbs = [0.5, 0.75]
            self.layer2CellUnits = 8
            self.layer2keepProb = 0.9

        elif scale == 'small':
            self.initialLearningRate = 0.001
            self.layer1CellUnits = [32, 16, 8]
            self.layer1keepProbs = [0.5, 0.7, 0.9]
            self.layer2CellUnits = 8
            self.layer2keepProb = 1.

        elif scale=='full':
            self.initialLearningRate = 0.001
            self.layer1CellUnits = [128, 128, 64, 32]
            self.layer1keepProbs = [0.5, 0.5, 0.6, 0.9]
            self.layer2CellUnits = 32
            self.layer2keepProb = 0.9


        assert len(self.layer1CellUnits) == len(self.layer1keepProbs)

        self.scale = scale
        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('config.network').info
        self.print()

    def print(self):
        self._logFunc('SHUFFLED %d hidden layer(s)' % len(self.layer1CellUnits))
        self._logFunc('LAYER1: number of LSTM cell units: ' + str(self.layer1CellUnits))
        self._logFunc('LAYER1: dropoutKeepProbs: ' + str(self.layer1keepProbs))
        self._logFunc('LAYER2: number of LSTM cell units: %d' % self.layer2CellUnits)
        self._logFunc('LAYER2: dropoutKeepProb: %0.3f' % self.layer2keepProb)


class Model2(object):
    def __init__(self, configScale_, input_, numClasses_, loggerFactory=None):
        """
        :type configScale_: string
        :type numClasses_: int
        :type input_: dict
        """
        assert configScale_ in ['basic', 'tiny', 'small', 'full']
        self.config = Model2Config(configScale_, loggerFactory)

        self._lr = tf.Variable(self.config.initialLearningRate, name='learningRate')
        x = input_['x']
        y = input_['y']
        numSeqs = input_['numSeqs']

        # ------ stack of LSTM - bi-directional RNN layer ------
        forwardCells = self.make_stacked_cells()
        backwardCells = self.make_stacked_cells()

        self.layer1outputs = tf.concat(
            tf.nn.bidirectional_dynamic_rnn(forwardCells, backwardCells,
                                            time_major=False, inputs=x, dtype=tf.float32,
                                            sequence_length=numSeqs,
                                            swap_memory=True)[0], 2)

        # self.layer1outputs = tf.nn.bidirectional_dynamic_rnn(forwardCells, backwardCells,
        #                                     time_major=False, inputs=x, dtype=tf.float32,
        #                                     sequence_length=numSeqs,
        #                                     swap_memory=True)[1][0]

        # ------ single RNN layer ------
        self.outputs = tf.nn.dynamic_rnn(
            DropoutWrapper(BasicLSTMCell(self.config.layer2CellUnits), output_keep_prob=self.config.layer2keepProb),
            time_major=False, dtype=tf.float32,
            inputs=self.layer1outputs, sequence_length=numSeqs)[0]

        self.output = last_relevant(self.outputs, numSeqs)

        # ------ final softmax layer ------
        weights = tf.Variable(tf.random_normal([self.config.layer2CellUnits, numClasses_]), name='weights')
        biases = tf.Variable(tf.random_normal([numClasses_]), name='biases')

        self.logits = tf.matmul(self.output, weights) + biases
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y))

        # ------ optimizer ------
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self.cost)

        # ------ metrics ------
        self.pred = tf.argmax(self.logits, 1)
        self.trueY = tf.argmax(y, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.trueY), tf.float32))

        summary.scalar('cost', self.cost)
        summary.scalar('accuracy', self.accuracy)
        self.merged_summaries = summary.merge_all()

    def make_stacked_cells(self):
        return MultiRNNCell([DropoutWrapper(BasicLSTMCell(f), output_keep_prob=k) if k < 1 else BasicLSTMCell(f)
                             for f, k in zip(self.config.layer1CellUnits, self.config.layer1keepProbs)])

    def lr(self, sess_):
        return sess_.run(self._lr)

    def assign_lr(self, sess_, newLearningRate_):
        sess_.run(tf.assign(self._lr, newLearningRate_))

    def train_op(self, sess_, feedDict_, computeMetrics_):
        thingsToRun = [self.optimizer] + [self.merged_summaries, self.cost, self.accuracy] if computeMetrics_ else []

        return sess_.run(thingsToRun, feedDict_)[1:]

    def evaluate(self, sess_, feedDict_):
        return sess_.run([self.cost, self.accuracy, self.trueY, self.pred], feedDict_)

if __name__ == '__main__':
    with tf.device('/cpu:0'):

        dr = DataReader('./data/peopleData/2_samples', 'bucketing', batchSize_=10)
        model = Model2('basic', dr.input, dr.numClasses)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        for step in range(10):
            _, c, acc = model.train_op(sess, dr.get_next_training_batch()[0], True)
            print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))