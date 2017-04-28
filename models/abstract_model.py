import tensorflow as tf
from tensorflow import name_scope, summary

from abc import ABCMeta, abstractmethod



class AbstractModel(metaclass=ABCMeta):

    def __init__(self, input_, loggerFactory_=None):
        """
        :type input_: dict
        """

        assert 'x' in input_ and 'y' in input_ and 'numSeqs' in input_

        self._lr = tf.Variable(self.initialLearningRate, name='learningRate')
        self.loggerFactory = loggerFactory_
        self.print = print if loggerFactory_ is None else loggerFactory_.getLogger('Model').info

        self.input = input_
        self.x = input_['x']
        self.y = input_['y']
        self.numSeqs = input_['numSeqs']
        self.vecDim = self.x.get_shape()[-1].value
        self.numClasses = self.y.get_shape()[-1].value
        self.outputs = []
        self.layers = []

        self.make_graph()

        with name_scope('predictions'):
            self.pred = tf.argmax(self.output, 1)
            self.trueY = tf.argmax(self.y, 1)

        with name_scope('metrics'):
            self.l2Loss = self.l2RegLambda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.output,
                    labels=self.y)) \
                        + self.l2Loss

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.trueY), tf.float32))

            summary.scalar('cost', self.cost)
            summary.scalar('accuracy', self.accuracy)

        with name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self.cost)

        self.merged_summaries = summary.merge_all()


    def lr(self, sess_):
        return sess_.run(self._lr)

    def assign_lr(self, sess_, newLearningRate_):
        assert newLearningRate_ > 0
        sess_.run(tf.assign(self._lr, newLearningRate_))

    def train_op(self, sess_, feedDict_, computeMetrics_):
        thingsToRun = [self.optimizer] + [self.merged_summaries, self.cost, self.accuracy] if computeMetrics_ else []

        return sess_.run(thingsToRun, feedDict_)[1:]

    def evaluate(self, sess_, feedDict_):
        return sess_.run([self.cost, self.accuracy, self.trueY, self.pred], feedDict_)

    @property
    def l2RegLambda(self):
        return self.__l2RegLambda

    @l2RegLambda.setter
    def l2RegLambda(self, val):
        assert val >= 0
        self.__l2RegLambda = val

    @property
    def initialLearningRate(self):
        return self.__initialLearningRate

    @initialLearningRate.setter
    def initialLearningRate(self, val):
        assert val > 0
        self.__initialLearningRate = val

    @property
    def output(self):
        return self.layers[-1].output

    # @output.setter
    # def output(self, val):
    #     self.__output = val

    @abstractmethod
    def make_graph(self):
        raise NotImplementedError('This (%s) is an abstract base class. Use one of its implementations instead.' % self.__class__.__name__)
