import tensorflow as tf
from tensorflow import name_scope, summary



class AbstractModel(object):

    def __init__(self, input_, initialLearningRate_, l2RegLambda_, loggerFactory_=None):
        """
        :type input_: dict
        :type initialLearningRate_: float
        :type l2RegLambda_: float
        """

        assert initialLearningRate_ > 0
        assert l2RegLambda_ > 0
        assert 'x' in input_ and 'y' in input_ and 'numSeqs' in input_

        self.l2RegLambda = l2RegLambda_
        self._lr = tf.Variable(initialLearningRate_, name='learningRate')
        self.loggerFactory = loggerFactory_
        self.print = print if loggerFactory_ is None else loggerFactory_.getLogger('Model').info

        self.input = input_
        self.x = input_['x']
        self.y = input_['y']
        self.numSeqs = input_['numSeqs']
        self.numClasses = self.y.get_shape()[-1].value

        self.output = self.make_graph()

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

    def make_graph(self):
        raise NotImplementedError('This (%s) is an abstract base class. Use one of its implementations instead.' % self.__class__.__name__)

    def lr(self, sess_):
        return sess_.run(self._lr)

    def assign_lr(self, sess_, newLearningRate_):
        sess_.run(tf.assign(self._lr, newLearningRate_))

    def train_op(self, sess_, feedDict_, computeMetrics_):
        thingsToRun = [self.optimizer] + [self.merged_summaries, self.cost, self.accuracy] if computeMetrics_ else []

        return sess_.run(thingsToRun, feedDict_)[1:]

    def evaluate(self, sess_, feedDict_):
        return sess_.run([self.cost, self.accuracy, self.trueY, self.pred], feedDict_)
