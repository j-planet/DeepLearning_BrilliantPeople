import tensorflow as tf
from tensorflow.contrib.learn import preprocessing
import json, os, glob
from pprint import pformat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from abc import ABCMeta, abstractmethod



def one_hot(ind, vecLen):
    res = [0] * vecLen
    res[ind] = 1

    return np.array(res)

def patch_arrays(arrays, lengths, numrows=None):
    """
    patch all arrays to have the same number of rows
    :param numrows: if None, patch to the max number of rows in the arrays
    :param arrays: 
    :return:  
    """

    assert len(arrays) == len(lengths)

    # pad to the largest array
    padLen = lengths.max()
    assert numrows is None or numrows >= padLen, 'numrows is fewer than the max number of rows: %d vs %d.' % (numrows, padLen)
    padLen = numrows or padLen

    res = np.zeros( (len(arrays), padLen, arrays[0].shape[1]) )

    for i, arr in enumerate(arrays):
        res[i][:arr.shape[0], :] = arr

    return res

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


class AbstractDataReader(metaclass=ABCMeta):

    def __init__(self, inputSource, bucketingOrRandom, batchSize_, minimumWords,
                 loggerFactory=None, train_valid_test_split_=(0.8, 0.1, 0.1)):

        assert bucketingOrRandom=='bucketing' or bucketingOrRandom=='random'
        assert sum(train_valid_test_split_)==1. and np.all([v > 0 for v in train_valid_test_split_]), 'Invalid train-validation-test split values.'

        self.inputSource = inputSource
        self.minimumWords = minimumWords
        self.trainBatchIndex = 0
        self.print = loggerFactory.getLogger('DataReader').info if loggerFactory else print
        self._batchSize = batchSize_
        self._bucketingOrRandom = bucketingOrRandom
        self._train_valid_test_split = train_valid_test_split_

        self._read_data_from_files()  # extract word2vec from files

        self.x, self.y, self.numSeqs = self.setup_placeholders()


    def _read_data_from_files(self):

        XData, YData_raw_labels, xLengths, names = self._read_raw_data()

        # transform Y data into a one-hot matrix
        self.yEncoder = LabelEncoder()
        YData = self.yEncoder.fit_transform(YData_raw_labels) # just list of indices here
        self.classLabels = self.yEncoder.classes_
        self.numClasses = len(self.classLabels)
        YData = np.array([one_hot(v, len(self.classLabels)) for v in YData])

        # train-validation-test split
        indices = dict(zip(['train', 'valid', 'test'],
                           train_valid_test_split(YData_raw_labels, *self._train_valid_test_split, logFunc_=self.print)))

        # bucket or sort training data
        if self._bucketingOrRandom == 'bucketing':
            orders = np.argsort([len(d) for d in XData[indices['train']]])  # increasing order of number of tokens
        elif self._bucketingOrRandom == 'random':
            orders = list(range(len(indices['train'])))
            np.random.shuffle(orders)
        else:
            raise Exception('Invalid bucketingOrRandom option:', self._bucketingOrRandom)

        indices['train'] = indices['train'][orders]

        # put data into batches
        self.data = {i: self._put_data_into_batches(XData[ind], YData[ind], xLengths[ind], names[ind])
                     for i, ind in indices.items()}

        self.numBatches = {i: len(d) for i, d in self.data.items()}
        self.trainSize = len(indices['train'])

        for i, ind in indices.items():
            if self._batchSize > len(ind):
                self.print('NOTE: actual %s batch size (%d) is smaller than assigned (%d)' % (i, len(ind), self._batchSize))

        self.print('%d train batches, %d validation batches, %d test batches.' % (self.numBatches['train'], self.numBatches['valid'], self.numBatches['test']))

        del XData, YData, xLengths, names, indices  # I hope this is unnecessary. But not a lot of faith in Python's garbage-collection speed.

    @abstractmethod
    def _read_raw_data(self):
        raise NotImplementedError('This (%s) is an abstract class.' % self.__class__.__name__)

    @abstractmethod
    def _put_data_into_batches(self, xData_, yData_, xLengths_, names_):
        raise NotImplementedError('This (%s) is an abstract class.' % self.__class__.__name__)

    @abstractmethod
    def setup_placeholders(self):
        raise NotImplementedError('This (%s) is an abstract class.' % self.__class__.__name__)

    def start_batch_from_beginning(self):
        self.trainBatchIndex = 0

    def wherechu_at(self):
        return self.trainBatchIndex

    def get_next_training_batch(self, shuffle=False):
        """
        :type shuffle: bool 
        :return: feedict, names
        """

        x, y, xlengths, names = self.data['train'][self.trainBatchIndex]

        if shuffle:
            orders = np.random.permutation(len(x))
            np.take(x, orders, axis=0, out=x)
            np.take(y, orders, axis=0, out=y)
            np.take(xlengths, orders, out=xlengths)
            np.take(names, orders, out=names)

        self.trainBatchIndex = (self.trainBatchIndex + 1) % self.numBatches['train']

        return {self.x: x, self.y: y, self.numSeqs: xlengths}, names

    def get_validation_data_in_batches(self):
        for x, y, xlengths, names in self.data['valid']:
            yield {self.x: x, self.y: y, self.numSeqs: xlengths}, names

    def get_test_data_in_batches(self):
        for x, y, xlengths, names in self.data['test']:
            yield {self.x: x, self.y: y, self.numSeqs: xlengths}, names

    @property
    def input(self):
        return {'x': self.x, 'y': self.y, 'numSeqs': self.numSeqs}


class DataReader_Embeddings(AbstractDataReader):

    def __init__(self, inputFilesDir, bucketingOrRandom, batchSize_, minimumWords,
                 loggerFactory=None, train_valid_test_split_=(0.8, 0.1, 0.1)):

        super().__init__(inputFilesDir, bucketingOrRandom, batchSize_, minimumWords,
                         loggerFactory, train_valid_test_split_)

    def setup_placeholders(self):

        # in the order of: x, y, numSeqs
        return tf.placeholder(tf.float32, [None, None, self.vectorDimension]), \
               tf.placeholder(tf.float32, [None, self.numClasses]), \
               tf.placeholder(tf.int32)

    def _read_raw_data(self):
        XData = []
        xLengths = []
        YData = []
        names = []

        self.print('======= Reading pre-made vector files... =======')
        self.print('Data source: ' + self.inputSource)

        numSkipped = 0
        for inputFilename in glob.glob(os.path.join(self.inputSource, '*.json')):

            with open(inputFilename, encoding='utf8') as ifile:
                d = json.load(ifile)

            mat = np.array(d['mat'])

            if len(mat) < self.minimumWords:
                self.print('Skipping %s because it has %d (<%d) words.' % (
                    os.path.basename(inputFilename), len(mat), self.minimumWords))
                numSkipped += 1
                continue

            XData.append(mat)
            occ = d['occupation']
            xLengths.append(mat.shape[0])
            YData.append(occ if type(occ) == str else occ[-1])
            names.append(os.path.basename(inputFilename).split('.json')[0])

        self.vectorDimension = XData[0].shape[1]
        self.maxXLen = max([d.shape[0] for d in XData])

        self.print('%d out of %d skipped' % (numSkipped, numSkipped + len(XData)))

        return np.array(XData), np.array(YData), np.array(xLengths), np.array(names)

    def _put_data_into_batches(self, xData_, yData_, xLengths_, names_):
        """
        :param xData_: 3D array of shape (number of arrays, sequences, vecDim)
        :return: a list of tuples [({x, y, xlengths, names}]
        """

        assert len(xData_) == len(xLengths_) == len(yData_) == len(names_)

        res = []
        total = len(xData_)

        startInds = list(range(0, total, self._batchSize))
        stopInds = startInds[1:] + [total]

        for start, stop in zip(startInds, stopInds):
            x = patch_arrays(xData_[start:stop], xLengths_[start:stop])
            res.append((x, yData_[start:stop], xLengths_[start:stop], names_[start:stop]))

        return res



class DataReader_Text(AbstractDataReader):

    def __init__(self, inputFilename, bucketingOrRandom, batchSize_, minimumWords,
                 loggerFactory=None, train_valid_test_split_=(0.8, 0.1, 0.1)):

        super().__init__(inputFilename, bucketingOrRandom, batchSize_, minimumWords,
                         loggerFactory, train_valid_test_split_)


    def setup_placeholders(self):

        # in the order of: x, y, numSeqs
        return tf.placeholder(tf.int32, [None, self.maxXLen]), \
               tf.placeholder(tf.float32, [None, self.numClasses]), \
               tf.placeholder(tf.int32)

    def _read_raw_data(self):

        XData = []
        xLengths = []
        YData = []
        names = []

        self.print('======= Reading pre-made vector files... =======')
        self.print('Data source: ' + self.inputSource)

        numSkipped = 0

        with open(self.inputSource, encoding='utf8') as ifile:

            for d in json.load(ifile):
                occ = d['occupation']
                content = d['content']
                numTokens = len(content.split(' '))
                name = d['name']

                if numTokens < self.minimumWords:
                    self.print('Skipping %s because it has %d (<%d) words.' % (name, numTokens, self.minimumWords))
                    numSkipped += 1
                    continue

                XData.append(content)
                xLengths.append(numTokens)
                YData.append(occ if type(occ) == str else occ[-1])
                names.append(d['name'])

        self.print('%d out of %d skipped' % (numSkipped, numSkipped + len(XData)))
        self.maxXLen = max(xLengths)

        self.vocabProcessor = preprocessing.VocabularyProcessor(self.maxXLen)
        XData = list(self.vocabProcessor.fit_transform(XData))
        self.vocabSize = len(self.vocabProcessor.vocabulary_)

        return np.array(XData), np.array(YData), np.array(xLengths), np.array(names)

    def _put_data_into_batches(self, xData_, yData_, xLengths_, names_):
        """
        :param xData_: 3D array of shape (number of arrays, sequences, vecDim)
        :return: a list of tuples [({x, y, xlengths, names}]
        """

        assert len(xData_) == len(xLengths_) == len(yData_) == len(names_)

        total = len(xData_)

        startInds = list(range(0, total, self._batchSize))
        stopInds = startInds[1:] + [total]

        return [(xData_[start:stop],
                 yData_[start:stop],
                 xLengths_[start:stop],
                 names_[start:stop])
                for start, stop in zip(startInds, stopInds)]


if __name__ == '__main__':
    # dataReader = DataReader_Embeddings('./data/peopleData/2_samples', 'bucketing', 5, 1)

    # dataReader = DataReader_Text('./data/peopleData/tokensfiles/all.json', 'bucketing', 5, 1)
    dataReader = DataReader_Text('./data/peopleData/tokensfiles/pol_sci.json', 'bucketing', 2, 1)

    for _ in range(10):
        d, names = dataReader.get_next_training_batch()

        print(d)
        print(names)
