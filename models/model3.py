import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import numpy as np
import tensorflow as tf
from tensorflow import summary
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, DropoutWrapper

from data_reader import DataReader
from utilities import last_relevant


class Model3Config(object):
    def __init__(self, scale=None, extraConfigDict={}, loggerFactory=None):
        assert scale in [None, 'basic', 'tiny', 'small', 'full']
        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('config.network').info

        if scale is None:
            self.data = {
                'initialLearningRate': None,

                'rnn_num_cell_units': None,
                'rnn_dropkeepprobs': None,

                'cnn_filter_widths': None,
                'cnn_num_features_per_filter': None,
                'cnn_dropkeepprob': None,

                'l2RegLambda': None
            }

        elif scale == 'basic':
            self.data = {
                'initialLearningRate': 0.002,

                'rnn_num_cell_units': [8],
                'rnn_dropkeepprobs': [1.],

                'cnn_filter_widths': [1],
                'cnn_num_features_per_filter': 2,
                'cnn_dropkeepprob': 1.,

                'l2RegLambda': 0
            }

        elif scale == 'tiny':
            self.data = {
                'initialLearningRate': 0.002,

                'rnn_num_cell_units': [32, 8],
                'rnn_dropkeepprobs': [0.5, 0.9],

                'cnn_filter_widths': [2],
                'cnn_num_features_per_filter': 2,
                'cnn_dropkeepprob': 1.,

                'l2RegLambda': 1e-3
            }

        elif scale == 'small':
            self.data = {
                'initialLearningRate': 0.002,

                'rnn_num_cell_units': [32, 16, 8],
                'rnn_dropkeepprobs': [0.5, 0.7, 0.9],

                'cnn_filter_widths': [1, 2],
                'cnn_num_features_per_filter': 4,
                'cnn_dropkeepprob': 0.9,

                'l2RegLambda': 1e-4
            }

        elif scale=='full':
            self.data = {
                'initialLearningRate': 0.001,

                'rnn_num_cell_units': [256, 128, 32, 32],
                'rnn_dropkeepprobs': [1]*4,

                'cnn_filter_widths': [1, 2, 3],
                'cnn_num_features_per_filter': 64,
                'cnn_dropkeepprob': 1,

                'l2RegLambda': 5e-4
            }


        assert len(self.data['rnn_num_cell_units']) == len(self.data['rnn_dropkeepprobs'])
        assert np.all([v is not None for v in self.data.values()]), 'Not all configs are provided.'

        for k, v in extraConfigDict.items():
            if k in self.data:
                self.data[k] = v
            else:
                self._logFunc('%s is not configurable. Skipping...' % k)


        self.configKeys = self.data.keys()

        self.print()

    def print(self):
        for k, v in self.data.items(): self._logFunc(k + ': ' + str(v))


class Model3(object):

    def __init__(self, config_, input_, numClasses_, loggerFactory=None):
        """
        :type config_: dict
        :type numClasses_: int
        :type input_: dict
        """
        # assert configScale_ in ['basic', 'tiny', 'small', 'full']
        # self.config = Model3Config(configScale_, loggerFactory)
        self.config = config_
        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('Model').info
        self._logFunc('Class: Model')
        # self._isTraining = True     # in order to turn off dropout during evaluation
        self._filterWidths = self.config['cnn_filter_widths']
        self._numFeaturesPerFilter = self.config['cnn_num_features_per_filter']
        self._cnnPooledDropoutKeep = self.config['cnn_dropkeepprob']

        self._l2RegLambda = self.config['l2RegLambda']

        self._lr = tf.Variable(self.config['initialLearningRate'], name='learningRate')
        x = input_['x']
        y = input_['y']
        numSeqs = input_['numSeqs']

        # ------ layer #1: stack of LSTM - bi-directional RNN layer ------
        self._logFunc('layer #1: stack of LSTM - bi-directional RNN layer')
        forwardCells = self.make_stacked_cells()
        backwardCells = self.make_stacked_cells()

        self.outputs = tf.concat(
            tf.nn.bidirectional_dynamic_rnn(forwardCells, backwardCells,
                                            time_major=False, inputs=x, dtype=tf.float32,
                                            sequence_length=numSeqs,
                                            swap_memory=True)[0], 2)
        self.layer1output = last_relevant(self.outputs, input_['numSeqs'])

        # ------ layer #2: CNN layer ------
        self._logFunc('layer #2: CNN layer')

        self.layer2input = tf.expand_dims(tf.expand_dims(self.layer1output, 1), -1) # changes 2x16 to 2x1x16x1
        self.pooled_outputs = []
        self.relus = []
        self.convs = []
        for filterWidth in self._filterWidths:
            with tf.name_scope('conv-maxpool-%d' % filterWidth):
                # the second parameter is shape[2] of the input
                filterShape = [filterWidth, 2 * self.config['rnn_num_cell_units'][-1], 1, self._numFeaturesPerFilter]
                # filterShape = [filterWidth, 1, 1, self._numFeaturesPerFilter]

                cnnSeqLen = 1
                cnnFilterMat = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name='W')
                cnnFilterBias = tf.Variable(tf.constant(0.1, shape=[self._numFeaturesPerFilter]), name='b')

                # padd along the number of sequences axis if not enough for the conv filter
                curInput = tf.tile(self.layer2input, [1, filterWidth, 1, 1]) if cnnSeqLen < filterWidth else self.layer2input
                conv = tf.nn.conv2d(curInput, cnnFilterMat, strides=[1,1,1,1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, cnnFilterBias), name='relu')
                self.convs.append(conv)
                self.relus.append(h)

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max(cnnSeqLen - filterWidth, 0) + 1, 1, 1],   # padding='VALID' may not work for small seq lens
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')

                self.pooled_outputs.append(pooled)

        self.numTotalCnnFilters = self._numFeaturesPerFilter * len(self._filterWidths)
        self.pooled_outputs_flat = tf.reshape(tf.concat(self.pooled_outputs, 3), [-1, self.numTotalCnnFilters])

        # dropout the CNN pooled results
        self.layer2output = tf.nn.dropout(self.pooled_outputs_flat, self._cnnPooledDropoutKeep)


        # ------ final softmax layer ------
        self._logFunc('final softmax layer')
        weights = tf.Variable(tf.random_normal([self.numTotalCnnFilters, numClasses_]), name='weights')
        biases = tf.Variable(tf.random_normal([numClasses_]), name='biases')
        self.logits = tf.nn.xw_plus_b(self.layer2output, weights, biases, name='logits')

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
        keepProbs = self.config['rnn_dropkeepprobs']

        return MultiRNNCell([DropoutWrapper(BasicLSTMCell(f), output_keep_prob=k)
                             for f, k in zip(self.config['rnn_num_cell_units'], keepProbs)])

    def lr(self, sess_):
        return sess_.run(self._lr)

    def assign_lr(self, sess_, newLearningRate_):
        sess_.run(tf.assign(self._lr, newLearningRate_))

    def train_op(self, sess_, feedDict_, computeMetrics_):
        # self._isTraining = True
        thingsToRun = [self.optimizer] + [self.merged_summaries, self.cost, self.accuracy] if computeMetrics_ else []

        return sess_.run(thingsToRun, feedDict_)[1:]

    def evaluate(self, sess_, feedDict_):
        # self._isTraining = False
        return sess_.run([self.cost, self.accuracy, self.trueY, self.pred], feedDict_)

if __name__ == '__main__':
    dr = DataReader('./data/peopleData/2_samples', 'bucketing', 20)
    config = Model3Config('tiny').data
    model = Model3(config, dr.input, dr.numClasses)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(10):
        _, c, acc = model.train_op(sess, dr.get_next_training_batch()[0], True)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))