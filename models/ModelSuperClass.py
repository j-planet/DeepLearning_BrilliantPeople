import tensorflow as tf
from tensorflow import name_scope, summary

from models.rnn_layer import RNNLayer
from data_reader import DataReader



class ModelBase(object):

    def __init__(self, input_, layerMakers_, initialLearningRate_, l2RegLambda_, loggerFactory_=None):
        """
        :type input_: dict
        :type initialLearningRate_: float
        :type l2RegLambda_: float
        """

        self.layerMakers = layerMakers_
        self.l2RegLambda = l2RegLambda_
        self._lr = tf.Variable(initialLearningRate_, name='learningRate')
        self.print = print if loggerFactory_ is None else loggerFactory_.getLogger('Model').info

        self._x = input_['x']
        self._y = input_['y']
        self._numSeqs = input_['numSeqs']

        prevInput = input_
        self.outputs = []

        for layerMaker in self.layerMakers:
            layer = layerMaker(prevInput)
            output = layer.output
            prevInput = output
            self.outputs.append((layer.__class__.__name__, output))

        with name_scope('predictions'):
            self.output = self.outputs[-1][1]
            self.pred = tf.argmax(self.output, 1)
            self.trueY = tf.argmax(self._y, 1)

        with name_scope('metrics'):
            self.l2Loss = self.l2RegLambda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.output,
                    labels=self._y)) \
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
        sess_.run(tf.assign(self._lr, newLearningRate_))

    def train_op(self, sess_, feedDict_, computeMetrics_):
        thingsToRun = [self.optimizer] + [self.merged_summaries, self.cost, self.accuracy] if computeMetrics_ else []

        return sess_.run(thingsToRun, feedDict_)[1:]

    def evaluate(self, sess_, feedDict_):
        return sess_.run([self.cost, self.accuracy, self.trueY, self.pred], feedDict_)


if __name__ == '__main__':
    dr = DataReader('../data/peopleData/2_samples', 'bucketing', 10)
    layer1 = lambda input_: RNNLayer(input_, [32, 16], [0.5, 1.], 1)

    model = ModelBase(dr.input, [layer1], 1e-3, 1e-4)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(10):
        _, c, acc = model.train_op(sess, dr.get_next_training_batch()[0], True)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))