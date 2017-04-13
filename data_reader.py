import json, os, glob
from pprint import pprint
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from data_processing.file2vec import file2vec, extract_embedding, extract_tokenset_from_file


def one_hot(ind, vecLen):
    res = [0] * vecLen
    res[ind] = 1

    return np.array(res)

def patch_arrays(arrays, numrows=None):
    """
    patch all arrays to have the same number of rows
    :numrows: if None, patch to the max number of rows in the arrays
    :param arrays: 
    :return:  
    """

    res = []

    lengths = [arr.shape[0] for arr in arrays]
    padLen = max(lengths)

    assert numrows is None or numrows >= padLen, 'numrows is fewer than the max number of rows: %d vs %d.' % (numrows, padLen)

    padLen = numrows or padLen


    for arr in arrays:
        res.append(np.append(arr, np.zeros((padLen - arr.shape[0], arr.shape[1])), axis=0))

    return np.array(res), lengths


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
        # sanity check that the stratified split worked properly
        print('train : validation : test = %d : %d : %d' % (len(train_indices), len(valid_indices), len(test_indices)))
        pprint(Counter(YData_[train_indices]))
        pprint(Counter(YData_[valid_indices]))
        pprint(Counter(YData_[test_indices]))

    return train_indices, valid_indices, test_indices


class DataReader(object):

    def _read_data_from_files(self, vectorFilesDir):

        # with open(os.path.join(peopleDataDir_, 'processed_names.json'), encoding='utf8') as ifile:
        #     self._peopleData = json.load(ifile)

        XData = []
        YData = []
        names = []

        print('======= Reading pre-made vector files... =======')

        for inputFilename in glob.glob(os.path.join(vectorFilesDir, '*.json')):

            with open(inputFilename, encoding='utf8') as ifile:
                d = json.load(ifile)

            XData.append(np.array(d['mat']))
            YData.append(d['occupation'])
            names.append(inputFilename.split('/')[-1].split('.')[0])

        self.XData = np.array(XData)
        self.YData_raw_labels = np.array(YData)
        self.maxXLen = max([d.shape[0] for d in self.XData])
        self.names = np.array(names)


    def __init__(self, vectorFilesDir=None):
        """
        
        :param vectorFilesDir: if files have already been converted to vectors, provide this directory. embeddingsFilename will be ignored 
        :param embeddingsFilename: if files have not been provided, convert files on the fly
        :param peopleDataDir_: 
        """

        self.globalBatchIndex = 0

        # extract word2vec from files
        self._read_data_from_files(vectorFilesDir)

        # transform Y data into a one-hot matrix
        self.yEncoder = LabelEncoder()
        self.YData = self.yEncoder.fit_transform(self.YData_raw_labels) # just list of indices here
        self.classLabels = self.yEncoder.classes_

        self.YData = np.array([one_hot(v, len(self.classLabels)) for v in self.YData])


        # ======= TRAIN-VALIDATION-TEST SPLIT=======

        self.train_indices, self.valid_indices, self.test_indices = train_valid_test_split(self.YData_raw_labels, 0.8, 0.1, 0.1)

        # self.XData_valid = self.XData[valid_indices]
        # self.YData_valid = self.YData[valid_indices]
        #
        # self.XData_test = self.XData[test_indices]
        # self.YData_test = self.YData[test_indices]


        # ======= BUCKETING TRAINING DATA =======
        # XData_train = self.XData[train_indices]
        # YData_train = self.YData[train_indices]

        # orders = np.argsort([len(d) for d in XData_train])    # increasing order of number of tokens

        # give up bucketing. try just random orders.
        # orders = list(range(len(XData_train)))
        # np.random.shuffle(orders)
        #
        # self.XData_train = XData_train[orders]
        # self.YData_train = YData_train[orders]


    def start_batch_from_beginning(self):
        self.globalBatchIndex = 0

    def wherechu_at(self):
        return self.globalBatchIndex

    def get_people_data(self):
        return self._peopleData

    def get_next_training_batch(self, batchSize_, patchTofull_=False, verbose_ = True):

        totalNumData = len(self.train_indices)

        if self.globalBatchIndex + batchSize_ <= totalNumData:
            newBatchIndex = self.globalBatchIndex + batchSize_
            batchIndices = list(range(self.globalBatchIndex, newBatchIndex))

        else:
            newBatchIndex = self.globalBatchIndex + batchSize_ - totalNumData
            batchIndices = list(range(self.globalBatchIndex, totalNumData)) + list(range(0, newBatchIndex))

        self.globalBatchIndex = newBatchIndex

        # randomize within a batch (does this actually make a difference...? Don't think so.)
        np.random.shuffle(batchIndices)

        # pad the x batch
        XBatch, xLengths = patch_arrays(self.XData[self.train_indices][batchIndices], self.maxXLen if patchTofull_ else None)
        YBatch = self.YData[self.train_indices][batchIndices]
        names = self.names[self.train_indices][batchIndices]

        if verbose_:
            print('Indices:', batchIndices, '--> # tokens:', [len(d) for d in XBatch], '--> Y values:', YBatch)

        return XBatch, YBatch, xLengths, names

    def get_all_training_data(self, patchTofull_=False):
        x, xlengths = patch_arrays(self.XData[self.train_indices], self.maxXLen if patchTofull_ else None)

        return x, self.YData[self.train_indices], xlengths, self.names[self.train_indices]

    def get_all_validation_data(self, patchTofull_=False):
        x, xlengths = patch_arrays(self.XData[self.valid_indices], self.maxXLen if patchTofull_ else None)

        return x, self.YData[self.valid_indices], xlengths, self.names[self.valid_indices]

    def get_validation_data_in_batches(self, batchSize_, patchTofull_=False):

        res = []

        total = len(self.valid_indices)
        numBatches = int(np.ceil(total/batchSize_))

        for i in range(numBatches):
            curIndices = self.valid_indices[i * batchSize_: (i+1)*batchSize_]

            x, xlengths = patch_arrays(self.XData[curIndices], self.maxXLen if patchTofull_ else None)

            res.append((x, self.YData[curIndices], xlengths, self.names[curIndices]))

        return res



    def get_all_test_data(self, patchTofull_=False):
        x, xlengths = patch_arrays(self.XData[self.test_indices], self.maxXLen if patchTofull_ else None)

        return x, self.YData[self.test_indices], xlengths, self.names[self.test_indices]

    def get_classes_labels(self):
        return self.classLabels

    def get_max_len(self):
        """
        :return: the maximum number of rows across ALL x's (including training, valid and testing) 
        """
        return self.maxXLen


if __name__ == '__main__':
    dataReader = DataReader(vectorFilesDir='./data/peopleData/earlyLifesWordMats')

    # x, y, xlengths = dataReader.get_next_training_batch(5)
    # dataReader.get_all_training_data()
    # dataReader.get_validation_data()
    # dataReader.get_test_data()