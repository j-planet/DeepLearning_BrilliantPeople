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


class ModelConfig(object):
    def __init__(self, scale, loggerFactory=None):
        assert scale in ['basic', 'tiny', 'small', 'full']

        if scale == 'basic':
            self.numHiddenLayerFeatures = [8]
            self.outputKeepProbs = [0.9]

        elif scale == 'tiny':
            self.numHiddenLayerFeatures = [32, 8]
            self.outputKeepProbs = [0.5, 0.9]

        elif scale == 'small':
            self.numHiddenLayerFeatures = [32, 16, 8]
            self.outputKeepProbs = [0.5, 0.7, 0.9]

        elif scale=='full':
            self.numHiddenLayerFeatures = [128, 128, 64, 32]
            self.outputKeepProbs = [0.5, 0.7, 0.8, 0.9]


        assert len(self.numHiddenLayerFeatures)==len(self.outputKeepProbs)

        self.scale = scale
        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('config.network').info
        self.print()

    def print(self):
        self._logFunc('SHUFFLED %d hidden layer(s)' % len(self.numHiddenLayerFeatures))
        self._logFunc('number of LSTM cell units: ' + str(self.numHiddenLayerFeatures))
        self._logFunc('dropoutKeepProbs: ' + str(self.outputKeepProbs))


class Model(object):
    def __init__(self, configScale_, input_, numClasses_, initialLearningRate_, loggerFactory=None):
        """
        :type configScale_: string
        :type numClasses_: int
        :type input_: dict
        """
        assert configScale_ in ['tiny', 'small', 'full']
        self.config = ModelConfig(configScale_, loggerFactory)

        self._lr = tf.Variable(initialLearningRate_, name='learningRate')
        x = input_['x']
        y = input_['y']
        numSeqs = input_['numSeqs']

        # ------ stack of LSTM - bi-directional RNN layer ------
        forwardCells = self.make_stacked_cells()
        backwardCells = self.make_stacked_cells()

        self.outputs = tf.concat(
            tf.nn.bidirectional_dynamic_rnn(forwardCells, backwardCells,
                                            time_major=False, inputs=x, dtype=tf.float32,
                                            sequence_length=numSeqs,
                                            swap_memory=True)[0], 2)

        # ------ final softmax layer ------
        weights = tf.Variable(tf.random_normal([2 * self.config.numHiddenLayerFeatures[-1], numClasses_]), name='weights')
        biases = tf.Variable(tf.random_normal([numClasses_]), name='biases')

        self.output = last_relevant(self.outputs, input_['numSeqs'])
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
                             for f, k in zip(self.config.numHiddenLayerFeatures, self.config.outputKeepProbs)])

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
    dr = DataReader('./data/peopleData/2_samples', 'bucketing')
    model = Model('tiny', dr.input, dr.numClasses, 0.001)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    validationData = dr.get_all_validation_data()[0]

    for step in range(10):
        _, c, acc = model.train_op(sess, dr.get_next_training_batch(20)[0], True)
        validC, validAcc, _, _ = model.evaluate(sess, validationData)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f), validation (%0.3f, %0.3f)' % (step, c, acc, validC, validAcc))