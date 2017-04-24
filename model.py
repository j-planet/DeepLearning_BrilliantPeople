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
            self.initialLearningRate = 0.002
            self.numHiddenLayerFeatures = [8]
            self.outputKeepProbs = [0.9]
            self.l2RegLambda = 1e-4

        elif scale == 'tiny':
            self.initialLearningRate = 0.002
            self.numHiddenLayerFeatures = [32, 8]
            self.outputKeepProbs = [1, 1]
            self.l2RegLambda = 1e-4

        elif scale == 'small':
            self.initialLearningRate = 0.002
            self.numHiddenLayerFeatures = [32, 16, 8]
            self.outputKeepProbs = [0.5, 0.7, 0.9]
            self.l2RegLambda = 1e-4

        elif scale=='full':
            self.initialLearningRate = 0.001
            self.numHiddenLayerFeatures = [256, 128, 32, 32]
            # self.outputKeepProbs = [0.5, 0.5, 0.9, 0.9]
            self.outputKeepProbs = [1.] * len(self.numHiddenLayerFeatures)
            self.l2RegLambda = 1e-5


        assert len(self.numHiddenLayerFeatures)==len(self.outputKeepProbs)

        self.scale = scale
        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('config.network').info
        self.print()

    def print(self):
        self._logFunc('SHUFFLED %d hidden layer(s)' % len(self.numHiddenLayerFeatures))
        self._logFunc('number of LSTM cell units: ' + str(self.numHiddenLayerFeatures))
        self._logFunc('dropoutKeepProbs: ' + str(self.outputKeepProbs))
        self._logFunc('initial learning rate: %0.3f' % self.initialLearningRate)
        self._logFunc('L2 penalty lambda: %f' % self.l2RegLambda)


class Model(object):

    def __init__(self, configScale_, input_, numClasses_, loggerFactory=None):
        """
        :type configScale_: string
        :type numClasses_: int
        :type input_: dict
        """
        assert configScale_ in ['basic', 'tiny', 'small', 'full']
        self.config = ModelConfig(configScale_, loggerFactory)
        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('Model').info
        self._logFunc('Class: Model')
        self._isTraining = True     # in order to turn off dropout during evaluation
        # self._l2RegLambda = self.config.l2RegLambdaPerClass * numClasses_
        self._l2RegLambda = self.config.l2RegLambda

        self._lr = tf.Variable(self.config.initialLearningRate, name='learningRate')
        x = input_['x']
        y = input_['y']
        numSeqs = input_['numSeqs']

        # ------ stack of LSTM - bi-directional RNN layer ------
        self._logFunc('stack of LSTM - bi-directional RNN layer')
        forwardCells = self.make_stacked_cells()
        backwardCells = self.make_stacked_cells()

        self.outputs = tf.concat(
            tf.nn.bidirectional_dynamic_rnn(forwardCells, backwardCells,
                                            time_major=False, inputs=x, dtype=tf.float32,
                                            sequence_length=numSeqs,
                                            swap_memory=True)[0], 2)

        # ------ final softmax layer ------
        self._logFunc('final softmax layer')
        weightShape = [2 * self.config.numHiddenLayerFeatures[-1], numClasses_]
        weights = tf.Variable(tf.random_normal(weightShape), name='weights')
        biases = tf.Variable(tf.random_normal([numClasses_]), name='biases')
        # self.l2Loss = self._l2RegLambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

        self.output = last_relevant(self.outputs, input_['numSeqs'])
        self.logits = tf.matmul(self.output, weights) + biases

        self.l2Loss = self._l2RegLambda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y)) + self.l2Loss

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
        # keepProbs = self.config.outputKeepProbs if self._isTraining else [1]*len(self.config.numHiddenLayerFeatures)
        keepProbs = self.config.outputKeepProbs

        return MultiRNNCell([DropoutWrapper(BasicLSTMCell(f), output_keep_prob=k)
                             for f, k in zip(self.config.numHiddenLayerFeatures, keepProbs)])

    def lr(self, sess_):
        return sess_.run(self._lr)

    def assign_lr(self, sess_, newLearningRate_):
        sess_.run(tf.assign(self._lr, newLearningRate_))

    def train_op(self, sess_, feedDict_, computeMetrics_):
        self._isTraining = True
        thingsToRun = [self.optimizer] + [self.merged_summaries, self.cost, self.accuracy] if computeMetrics_ else []

        return sess_.run(thingsToRun, feedDict_)[1:]

    def evaluate(self, sess_, feedDict_):
        self._isTraining = False
        return sess_.run([self.cost, self.accuracy, self.trueY, self.pred], feedDict_)

if __name__ == '__main__':
    dr = DataReader('./data/peopleData/2_samples', 'bucketing', 20)
    model = Model('tiny', dr.input, dr.numClasses)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(10):
        _, c, acc = model.train_op(sess, dr.get_next_training_batch()[0], True)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))