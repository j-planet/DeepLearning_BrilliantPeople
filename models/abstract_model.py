import tensorflow as tf
from tensorflow import name_scope, summary

from abc import ABCMeta, abstractmethod



class AbstractModel(metaclass=ABCMeta):

    def __init__(self, input_, initialLearningRate, loggerFactory_=None):
        """
        :type input_: dict
        :type initialLearningRate: float 
        """

        assert 'x' in input_ and 'y' in input_ and 'numSeqs' in input_
        assert initialLearningRate > 0

        self.initialLearningRate = initialLearningRate
        self._lr = tf.Variable(self.initialLearningRate, name='learningRate')
        self.loggerFactory = loggerFactory_
        self.print = print if loggerFactory_ is None else loggerFactory_.getLogger('Model').info
        self.print('initial learning rate: %0.7f' % initialLearningRate)

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
            # self.l2Loss = self.l2RegLambda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
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

    def add_output(self, output, outputShape):
        self.outputs.append({'output': output, 'output_shape': outputShape})

    def add_layer(self, layerMaker_, input_=None, inputDim_=None):
        if input_ is None: input_ = self.prevOutput
        inputDim_ = inputDim_ or self.prevOutputShape

        layer = layerMaker_(input_, inputDim_, self.loggerFactory)
        self.layers.append(layer)
        self.add_output(layer.output, layer.output_shape)

        return layer

    @property
    def l2RegLambda(self):
        return self.__l2RegLambda

    @l2RegLambda.setter
    def l2RegLambda(self, val):
        assert val >= 0
        self.__l2RegLambda = val

    @property
    def l2Loss(self):
        return self.__dict__.get('__l2loss', 0)

    @l2Loss.setter
    def l2Loss(self, val):
        self.__l2loss = val

    @property
    def output(self):
        # return self.layers[-1].output
        return self.outputs[-1]['output']

    @property
    def prevOutput(self):
        # return self.layers[-1].output
        return self.outputs[-1]['output']

    @property
    def prevOutputShape(self):
        # return self.layers[-1].output_shape

        return self.outputs[-1]['output_shape']

    @abstractmethod
    def make_graph(self):
        raise NotImplementedError('This (%s) is an abstract base class. Use one of its implementations instead.' % self.__class__.__name__)
