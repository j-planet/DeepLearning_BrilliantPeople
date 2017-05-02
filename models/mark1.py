import tensorflow as tf

from models.abstract_model import AbstractModel
from data_readers import DataReader_Embeddings
from layers.rnn_layer import RNNLayer
from layers.fully_connected_layer import FullyConnectedLayer


class Mark1(AbstractModel):

    def __init__(self, input_, initialLearningRate,
                 rnnNumCellUnits, rnnKeepProbs=1,
                 loggerFactory_=None):
        """        
        :type initialLearningRate: float 
        :type rnnNumCellUnits: list
        :type rnnKeepProbs: Union[list, float]
        """

        if type(rnnKeepProbs) == float:
            assert 0 < rnnKeepProbs <= 1
            rnnKeepProbs = [rnnKeepProbs] * len(rnnNumCellUnits)

        assert len(rnnNumCellUnits) == len(rnnNumCellUnits)

        self.rnnNumCellUnits = rnnNumCellUnits
        self.rnnKeepProbs = rnnKeepProbs

        super().__init__(input_, initialLearningRate, loggerFactory_)

    def make_graph(self):

        self.add_layers(RNNLayer.new(self.rnnNumCellUnits),
                        self.input,
                        (-1, -1, self.vecDim))

        self.add_layers(FullyConnectedLayer.new(self.numClasses))


if __name__ == '__main__':

    datadir = '../data/peopleData/2_samples'
    # datadir = '../data/peopleData/earlyLifesWordMats/politician_scientist'

    lr = 1e-3
    dr = DataReader_Embeddings(datadir, 'bucketing', 40, 30)
    model = Mark1(dr.input, lr, [16, 8], [.5, .5])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        if step % 10 == 0:
            print('Lowering learning rate to', lr)
            lr *= 0.9
            model.assign_lr(sess, lr)

        fd = dr.get_next_training_batch()[0]
        _, c, acc = model.train_op(sess, fd, True)
        print('Step %d: (cost, accuracy): training (%0.3f, %0.3f)' % (step, c, acc))