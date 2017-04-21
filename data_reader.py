import tensorflow as tf
import json, os, glob
from pprint import pformat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import Counter


def one_hot(ind, vecLen):
    res = [0] * vecLen
    res[ind] = 1

    return np.array(res)

def patch_arrays(arrays, numrows=None):
    """
    patch all arrays to have the same number of rows
    :param numrows: if None, patch to the max number of rows in the arrays
    :param arrays: 
    :return:  
    """
    print('patching')
    lengths = np.array([arr.shape[0] for arr in arrays])
    padLen = lengths.max()

    assert numrows is None or numrows >= padLen, 'numrows is fewer than the max number of rows: %d vs %d.' % (numrows, padLen)
    padLen = numrows or padLen

    res = np.empty( (len(lengths), padLen, arrays[0].shape[1]) )

    for i, arr in enumerate(arrays):
        res[i][:arr.shape[0], :] = arr

    print('done patching')

    return res, lengths

def train_valid_test_split(YData_, trainSize_, validSize_, testSize_, verbose_=True, logFunc_=None):
    """
    :return: train_indices, valid_indices, test_indices 
    """

    logFunc_ = logFunc_ or print

    totalLen = len(YData_)

    # convert all lenghts to floats
    if type(trainSize_)==int: trainSize_ /= 1. * totalLen
    if type(validSize_)==int: validSize_ /= 1. * totalLen
    if type(testSize_)==int: testSize_ /= 1. * totalLen

    assert trainSize_ + validSize_ + testSize_ == 1, \
        'Sizes do not add up to 1: ' + trainSize_ + ' ' + validSize_ + ' ' + testSize_

    sss = StratifiedShuffleSplit(n_splits=1, test_size=testSize_, train_size=trainSize_, random_state=0)
    s = sss.split([None]*totalLen, YData_)
    train_indices, test_indices = list(s)[0]
    valid_indices = np.array([i for i in range(totalLen) if i not in train_indices and i not in test_indices])

    if verbose_:

        # sanity check that the stratified split worked properly
        logFunc_('train : validation : test = %d : %d : %d' % (len(train_indices), len(valid_indices), len(test_indices)))
        logFunc_(pformat(Counter(YData_[train_indices])))
        logFunc_(pformat(Counter(YData_[valid_indices])))
        logFunc_(pformat(Counter(YData_[test_indices])))

    return train_indices, valid_indices, test_indices


class DataReader(object):

    def __init__(self, vectorFilesDir, bucketingOrRandom, batchSize_,
                 loggerFactory=None, train_valid_test_split_=(0.8, 0.1, 0.1)):
        """
        :param vectorFilesDir: if files have already been converted to vectors, provide this directory. embeddingsFilename will be ignored
        :param bucketingOrRandom: one of {'bucketing', 'random'} for training data points order
        """

        assert bucketingOrRandom=='bucketing' or bucketingOrRandom=='random'
        assert sum(train_valid_test_split_)==1. and np.all([v > 0 for v in train_valid_test_split_]), 'Invalid train-validation-test split values.'

        self.trainBatchIndex = 0
        self._logFunc = loggerFactory.getLogger('DataReader').info if loggerFactory else print
        self._batchSize = batchSize_
        self._bucketingOrRandom = bucketingOrRandom
        self._train_valid_test_split = train_valid_test_split_

        # extract word2vec from files and split
        self._read_data_from_files(vectorFilesDir)  # extract word2vec from files

        # define Tensorflow input Placeholders
        self.x = tf.placeholder(tf.float32, [None, None, self.vectorDimension])
        self.y = tf.placeholder(tf.float32, [None, self.numClasses])
        self.numSeqs = tf.placeholder(tf.int32)


    def _read_data_from_files(self, vectorFilesDir):

        XData = []
        YData = []
        names = []

        self._logFunc('======= Reading pre-made vector files... =======')
        self._logFunc('Data source: ' + vectorFilesDir)

        for inputFilename in glob.glob(os.path.join(vectorFilesDir, '*.json')):

            with open(inputFilename, encoding='utf8') as ifile:
                d = json.load(ifile)

            XData.append(np.array(d['mat']))
            occ = d['occupation']
            YData.append(occ if type(occ)==str else occ[-1])
            names.append(os.path.basename(inputFilename).split('.json')[0])

        XData = np.array(XData)
        YData_raw_labels = np.array(YData)
        self.maxXLen = max([d.shape[0] for d in XData])
        names = np.array(names)
        self.vectorDimension = XData[0].shape[1]

        # transform Y data into a one-hot matrix
        self.yEncoder = LabelEncoder()
        YData = self.yEncoder.fit_transform(YData_raw_labels) # just list of indices here
        self.classLabels = self.yEncoder.classes_
        self.numClasses = len(self.classLabels)
        YData = np.array([one_hot(v, len(self.classLabels)) for v in YData])

        # train-validation-test split
        train_indices, valid_indices, test_indices = \
            train_valid_test_split(YData_raw_labels, *self._train_valid_test_split, logFunc_=self._logFunc)

        # bucket or sort training data
        if self._bucketingOrRandom == 'bucketing':
            orders = np.argsort([len(d) for d in XData[train_indices]])  # increasing order of number of tokens
        elif self._bucketingOrRandom == 'random':
            orders = list(range(len(train_indices)))
            np.random.shuffle(orders)
        else:
            raise Exception('Invalid bucketingOrRandom option:', self._bucketingOrRandom)

        train_indices = train_indices[orders]

        # put data into batches
        self.trainData = self._get_data_in_batches(XData[train_indices], YData[train_indices], names[train_indices])
        self.validData = self._get_data_in_batches(XData[valid_indices], YData[valid_indices], names[valid_indices])
        self.testData = self._get_data_in_batches(XData[test_indices], YData[test_indices], names[test_indices])

        self.numTrainBatches = len(self.trainData)
        self.numValidBatches = len(self.validData)
        self.numTestBatches = len(self.testData)
        self.trainSize = len(train_indices)

        if self._batchSize > len(train_indices): self._logFunc('NOTE: actual training batch size (%d) is smaller than assigned (%d)' % (len(train_indices), self._batchSize))
        if self._batchSize > len(valid_indices): self._logFunc('NOTE: actual validation batch size (%d) is smaller than assigned (%d)' % (len(valid_indices), self._batchSize))
        if self._batchSize > len(test_indices): self._logFunc('NOTE: actual test batch size (%d) is smaller than assigned (%d)' % (len(test_indices), self._batchSize))
        self._logFunc('%d train batches, %d validation batches, %d test batches.' % (self.numTrainBatches, self.numValidBatches, self.numTestBatches))

        del XData, YData, names, train_indices, valid_indices, test_indices


    def start_batch_from_beginning(self):
        self.trainBatchIndex = 0

    def wherechu_at(self):
        return self.trainBatchIndex

    def get_next_training_batch(self, shuffle=False):
        """
        :type shuffle: bool 
        :return: feedict, names
        """

        x, y, xlengths, names = self.trainData[self.trainBatchIndex]

        if shuffle:
            orders = np.random.permutation(len(x))
            np.take(x, orders, axis=0, out=x)
            np.take(y, orders, axis=0, out=y)
            np.take(xlengths, orders, out=xlengths)
            np.take(names, orders, out=names)

        self.trainBatchIndex = (self.trainBatchIndex + 1) % self.numTrainBatches

        return {self.x: x, self.y: y, self.numSeqs: xlengths}, names

    def get_validation_data_in_batches(self):
        for x, y, xlengths, names in self.validData:
            yield {self.x: x, self.y: y, self.numSeqs: xlengths}, names

    def get_test_data_in_batches(self):
        for x, y, xlengths, names in self.testData:
            yield {self.x: x, self.y: y, self.numSeqs: xlengths}, names

    def _get_data_in_batches(self, xData_, yData_, names_):
        """
        :param xData_: 3D array of shape (number of arrays, sequences, vecDim)
        :return: a list of tuples [({x, y, xlengths, names}]
        """

        assert len(xData_) == len(yData_) == len(names_)

        res = []
        total = len(xData_)

        startInds = list(range(0, total, self._batchSize))
        stopInds = startInds[1:] + [total]

        for start, stop in zip(startInds, stopInds):
            x, lengths = patch_arrays(xData_[start:stop])
            res.append((x, yData_[start:stop], lengths, names_[start:stop]))

        return res

    @property
    def input(self):
        return {'x': self.x, 'y': self.y, 'numSeqs': self.numSeqs}

if __name__ == '__main__':
    dataReader = DataReader('./data/peopleData/2_samples', 'bucketing', batchSize_=5)

    for _ in range(10):
        d, names = dataReader.get_next_training_batch()

        print(d)
        print(names)
