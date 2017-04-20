from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
# import numpy as np
import tensorflow as tf
# from tensorflow import summary

from data_reader import DataReader
from model import Model
from utilities import tensorflowFilewriters, label_comparison, LoggerFactory, create_time_dir, dir_create_n_clear


# sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)))
sess = tf.InteractiveSession()


def evaluate_in_batches(data_, yLabelTexts_, evaluationFunc_, logFunc_=None):
    """
    :param data_: a list of [(feeddict, names),...] 
    :type data_: list
    """
    logFunc_ = logFunc_ or print

    # data = dataReader.get_data_in_batches(batchSize_, validateOrTest_)

    totalCost = 0
    totalAccuracy = 0
    allTrueYInds = []
    allPredYInds = []
    allNames = []

    # d: x, y, xLengths, names
    for fd, names in data_:
        c, acc, trueYInds, predYInds = evaluationFunc_(sess, fd)

        actualCount = len(trueYInds)
        totalCost += c * actualCount
        totalAccuracy += acc * actualCount
        allNames += list(names)
        allTrueYInds += list(trueYInds)
        allPredYInds += list(predYInds)

    assert len(allTrueYInds) == len(allPredYInds) == len(allNames)

    avgCost = totalCost / len(allTrueYInds)
    avgAccuracy = totalAccuracy / len(allTrueYInds)

    logFunc_('-------------')
    logFunc_('loss = %.3f, accuracy = %.3f' % (avgCost, avgAccuracy))
    label_comparison(allTrueYInds, allPredYInds, allNames, yLabelTexts_, logFunc_)
    logFunc_('-------------')



class RunConfig(object):
    def __init__(self, scale, loggerFactory=None):
        assert scale in ['basic', 'tiny', 'small', 'full']

        if scale == 'basic':
            self.initialLearningRate = 0.001
            self.numSteps = 5  # 1 step runs 1 batch
            self.batchSize = 10
            self.logValidationEvery = 3

        elif scale == 'tiny':
            self.initialLearningRate = 0.003
            self.numSteps = 10  # 1 step runs 1 batch
            self.batchSize = 20
            self.logValidationEvery = 2

        elif scale == 'small':
            self.initialLearningRate = 0.002
            self.numSteps = 100  # 1 step runs 1 batch
            self.batchSize = 50
            self.logValidationEvery = 10

        elif scale == 'full':
            self.initialLearningRate = 0.002
            self.numSteps = 500  # 1 step runs 1 batch
            self.batchSize = 150
            self.logValidationEvery = 20

        self.scale = scale
        self._logFunc = print if loggerFactory is None else loggerFactory.getLogger('config.run').info
        self.print()

    def print(self):
        self._logFunc('batch size %d, initial learning rate %.3f' % (self.batchSize, self.initialLearningRate))


def log_progress(step, numDataPoints, lr, c=None, acc=None, logFunc=None):

    res = 'Step %d (%d data pts); lr = %0.4f' % (step, numDataPoints, lr)

    if c is not None: res += '; loss = %.3f' % c
    if acc is not None: res += ', accuracy = %.3f' % acc

    (logFunc or print)(res)

    return res


def evaluate_stored_model(dataDir, modelScale, savePath):

    dataReader = DataReader(dataDir, 'bucketing')
    model = Model(modelScale, dataReader.input, dataReader.numClasses, 0)

    # sess.run(tf.global_variables_initializer())
    tf.train.Saver().restore(sess, savePath)

    evaluate_in_batches(dataReader.get_data_in_batches(10, 'test'), dataReader.classLabels, model.evaluate)


def main(dataDir, modelScale, runScale, logDir=create_time_dir('./logs/main')):
    assert modelScale in ['basic', 'tiny', 'small', 'full']
    assert runScale in ['basic', 'tiny', 'small', 'full']

    def _decrease_learning_rate(numDataPoints_, decayPerCycle_=0.95, lowerBound_=1e-4):
        """
        :type numDataPoints_: int
        """

        lrDecay = decayPerCycle_ ** (numDataPoints_ / dataReader.trainSize)
        newLr = max(runConfig.initialLearningRate * lrDecay, lowerBound_)
        model.assign_lr(sess, newLr)

        return newLr

    st = time()
    loggerFactory = LoggerFactory(logDir)
    runConfig = RunConfig(runScale, loggerFactory)
    dataReader = DataReader(dataDir, 'bucketing', loggerFactory)

    model = Model(modelScale, dataReader.input, dataReader.numClasses, runConfig.initialLearningRate, loggerFactory)

    # =========== set up tensorboard ===========
    train_writer, valid_writer = tensorflowFilewriters(logDir)
    train_writer.add_graph(sess.graph)
    trainLogFunc = loggerFactory.getLogger('run.train').info
    validLogFunc = loggerFactory.getLogger('run.validate').info
    testLogFunc = loggerFactory.getLogger('run.test').info

    # =========== TRAIN!! ===========
    sess.run(tf.global_variables_initializer())
    saver, savePath = tf.train.Saver(), os.path.join(dir_create_n_clear(logDir, 'saved'), 'save.ckpt')
    trainLogFunc('Saving to ' + savePath)
    dataReader.start_batch_from_beginning()     # technically unnecessary
    batchSize, numSteps, logValidationEvery = runConfig.batchSize, runConfig.numSteps, runConfig.logValidationEvery


    for step in range(numSteps):
        numDataPoints = (step+1) * runConfig.batchSize

        lr = _decrease_learning_rate(numDataPoints)
        summaries, c, acc = model.train_op(sess, dataReader.get_next_training_batch(batchSize)[0], computeMetrics_=True)

        train_writer.add_summary(summaries, step * batchSize)

        log_progress(step, numDataPoints, lr, c, acc, trainLogFunc)

        if step % logValidationEvery == 0:
            evaluate_in_batches(dataReader.get_data_in_batches(10, 'validate'), dataReader.classLabels, model.evaluate, validLogFunc)
            saver.save(sess, savePath, global_step=numDataPoints)


    testLogFunc('Time elapsed: ' + str(time()-st))
    evaluate_in_batches(dataReader.get_data_in_batches(10, 'test'), dataReader.classLabels, model.evaluate, testLogFunc)

    saver.save(sess, savePath)
    train_writer.close()
    valid_writer.close()

if __name__ == '__main__':
    DATA_DIRs = {'tiny_fake': './data/peopleData/2_samples',
                 'small_2occupations': './data/peopleData/earlyLifesWordMats/politician_scientist',
                 'small': './data/peopleData/earlyLifesWordMats',
                 'full_2occupations': './data/peopleData/earlyLifesWordMats_42B300d/politician_scientist',
                 'full': './data/peopleData/earlyLifesWordMats_42B300d'}

    # with tf.device('/cpu:0'):
    main(DATA_DIRs['tiny_fake'], 'tiny', 'tiny', create_time_dir('./logs/main'))

    # evaluate_stored_model(DATA_DIRs['tiny_fake'], 'tiny', './logs/main/04202017 01:12:26/saved/save.ckpt')