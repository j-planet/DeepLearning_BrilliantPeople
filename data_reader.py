import tensorflow as tf
import json, os, glob
from pprint import pformat
import logging
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
    lengths = [arr.shape[0] for arr in arrays]
    padLen = max(lengths)

    assert numrows is None or numrows >= padLen, 'numrows is fewer than the max number of rows: %d vs %d.' % (numrows, padLen)
    padLen = numrows or padLen

    res = np.empty( (len(lengths), padLen, arrays[0].shape[1]) )

    for i, arr in enumerate(arrays):
        res[i][:arr.shape[0], :] = arr

    print('done patching')

    return res, lengths

def train_valid_test_split(YData_, trainSize_, validSize_, testSize_, verbose_=True):
    """
    :return: train_indices, valid_indices, test_indices 
    """

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
        logger = logging.getLogger('DataReader')

        # sanity check that the stratified split worked properly
        logger.info('train : validation : test = %d : %d : %d' % (len(train_indices), len(valid_indices), len(test_indices)))
        logger.info(pformat(Counter(YData_[train_indices])))
        logger.info(pformat(Counter(YData_[valid_indices])))
        logger.info(pformat(Counter(YData_[test_indices])))

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

        self.globalBatchIndex = 0
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

        self.XData = np.array(XData)
        self.YData_raw_labels = np.array(YData)
        self.maxXLen = max([d.shape[0] for d in self.XData])
        self.names = np.array(names)
        self.vectorDimension = self.XData[0].shape[1]

        # transform Y data into a one-hot matrix
        self.yEncoder = LabelEncoder()
        self.YData = self.yEncoder.fit_transform(self.YData_raw_labels) # just list of indices here
        self.classLabels = self.yEncoder.classes_
        self.numClasses = len(self.classLabels)
        self.YData = np.array([one_hot(v, len(self.classLabels)) for v in self.YData])

        # train-validation-test split
        self.train_indices, self.valid_indices, self.test_indices = \
            train_valid_test_split(self.YData_raw_labels, *self._train_valid_test_split)

        self.trainSize = len(self.train_indices)
        self.validSize = len(self.valid_indices)
        self.testSize = len(self.test_indices)

        # bucket or sort training data
        if self._bucketingOrRandom == 'bucketing':
            orders = np.argsort([len(d) for d in self.XData[self.train_indices]])  # increasing order of number of tokens
        elif self._bucketingOrRandom == 'random':
            orders = list(range(len(self.train_indices)))
            np.random.shuffle(orders)
        else:
            raise Exception('Invalid bucketingOrRandom option:', self._bucketingOrRandom)

        self.train_indices = self.train_indices[orders]

        # put data into batches


    def start_batch_from_beginning(self):
        self.globalBatchIndex = 0

    def wherechu_at(self):
        return self.globalBatchIndex

    def get_next_training_batch(self, batchSize_, patchTofull_=False, verbose_ = False):

        totalNumData = len(self.train_indices)

        if self.globalBatchIndex + batchSize_ <= totalNumData:
            newBatchIndex = self.globalBatchIndex + batchSize_
            batchIndices = list(range(self.globalBatchIndex, newBatchIndex))

        else:
            temp = self.globalBatchIndex + batchSize_
            newBatchIndex = temp % totalNumData
            numRounds = int(temp/totalNumData)-1
            batchIndices = list(range(self.globalBatchIndex, totalNumData)) + list(range(totalNumData))*numRounds + list(range(newBatchIndex))
            if numRounds > 0:
                self._logFunc('Batch size %d > data size %d. Going around %d times from index %d to %d.' % (batchSize_, totalNumData, numRounds, self.globalBatchIndex, newBatchIndex))

        self.globalBatchIndex = newBatchIndex

        # randomize within a batch (does this actually make a difference...? Don't think so.)
        np.random.shuffle(batchIndices)

        # pad the x batch
        XBatch, xLengths = patch_arrays(self.XData[self.train_indices][batchIndices], self.maxXLen if patchTofull_ else None)
        YBatch = self.YData[self.train_indices][batchIndices]
        names = self.names[self.train_indices][batchIndices]

        if verbose_:
            self._logFunc('Indices:', batchIndices, '--> # tokens:', [len(d) for d in XBatch], '--> Y values:', YBatch)

        return {self.x: XBatch, self.y: YBatch, self.numSeqs: xLengths}, names

    def get_all_training_data(self, patchTofull_=False):
        x, xlengths = patch_arrays(self.XData[self.train_indices], self.maxXLen if patchTofull_ else None)

        return {self.x: x,
                self.y: self.YData[self.train_indices],
                self.numSeqs: xlengths}, self.names[self.train_indices]

    def get_all_validation_data(self, patchTofull_=False):
        x, xlengths = patch_arrays(self.XData[self.valid_indices], self.maxXLen if patchTofull_ else None)

        return {self.x: x,
                self.y: self.YData[self.valid_indices],
                self.numSeqs: xlengths}, self.names[self.valid_indices]

    def get_all_test_data(self, patchTofull_=False):
        x, xlengths = patch_arrays(self.XData[self.test_indices], self.maxXLen if patchTofull_ else None)

        return {self.x: x,
                self.y: self.YData[self.test_indices],
                self.numSeqs: xlengths}, self.names[self.test_indices]

    def _get_data_in_batches(self, data_, patchTofull_=False):
        """
        :param data_: 3D array of shape (number of arrays, sequences, vecDim)
        :return: a list
        """

        bs = self._batchSize
        total = len(data_)
        numLeftover = total % bs

        res = [patch_arrays(arr) for arr in np.split(data_[:(total - numLeftover)], bs) + (data_[-numLeftover:] if numLeftover>0 else [])]




        assert validTestOrTrain in ['validate', 'test', 'train']

        dataIndices = self.valid_indices if validTestOrTrain=='validate' else \
            self.test_indices if validTestOrTrain=='test' else self.train_indices

        res = []

        # total = len(dataIndices)
        numBatches = int(np.ceil(total / self._batchSize))

        for i in range(numBatches):
            curIndices = dataIndices[i * batchSize_: (i + 1) * batchSize_]

            x, xlengths = patch_arrays(self.XData[curIndices], self.maxXLen if patchTofull_ else None)

            res.append(({ self.x: x,
                          self.y: self.YData[curIndices],
                          self.numSeqs: xlengths
                          }, self.names[curIndices]))

        return res

    @property
    def input(self):
        return {'x': self.x, 'y': self.y, 'numSeqs': self.numSeqs}

if __name__ == '__main__':
    dataReader = DataReader('./data/peopleData/2_samples', 'bucketing')
    d, names = dataReader.get_next_training_batch(5)

    print(d)
    print(names)
