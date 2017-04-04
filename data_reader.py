import json, os
from pprint import pprint
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

from data_processing.extract_tokens import file2vec, extract_embedding, extract_tokenset_from_file


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

    def __init__(self, embeddingsFilename, peopleDataDir_ ='./data/peopleData'):

        self.globalBatchIndex = 0

        # ======= EXTRACT EMBEDDING & WORD2VEC =======
        print('======= Extracting embedding...')
        EMBEDDINGS, _ = extract_embedding(
            embeddingsFilename_= embeddingsFilename,
            relevantTokens_=extract_tokenset_from_file(peopleDataDir_.strip('/') + '/earlyLifeCorpus.txt'),
            includeUnk_=True,
            verbose=False
        )

        with open(peopleDataDir_.strip('/') + '/processed_names.json', encoding='utf8') as ifile:
            self._peopleData = json.load(ifile)

        XData = []
        YData = []
        nonexist = 0

        for name, d in self._peopleData.items():
            occupation = d['occupation'][-1]
            filename = './data/peopleData/earlyLifes/%s.txt' % name

            if os.path.exists(filename):
                mat = file2vec(filename, EMBEDDINGS)
                XData.append(mat)
                YData.append(occupation)

            else:
                print(filename, 'does not exist.')
                nonexist += 1

        self.XData = np.array(XData)
        self.YData = np.array(YData)


        # ======= TRAIN-VALIDATION-TEST SPLIT=======

        print('%d / %d do not exist.' % (nonexist, len(self._peopleData)))
        train_indices, valid_indices, test_indices = train_valid_test_split(self.YData, 0.7, 0.15, 0.15)


        self.XData_valid = self.XData[valid_indices]
        self.YData_valid = self.YData[valid_indices]

        self.XData_test = self.XData[test_indices]
        self.YData_test = self.YData[test_indices]


        # ======= BUCKETING TRAINING DATA =======
        XData_train = self.XData[train_indices]
        YData_train = self.YData[train_indices]

        orders = np.argsort([len(d) for d in XData_train])    # increasing order of number of tokens
        self.XData_train = XData_train[orders]
        self.YData_train = YData_train[orders]


    def start_batch_from_beginning(self):
        self.globalBatchIndex = 0

    def wherechu_at(self):
        return self.globalBatchIndex

    def get_raw_data(self):
        return self._peopleData

    def get_next_training_batch(self, batchSize_, verbose_ = True):

        totalNumData = len(self.XData_train)

        if self.globalBatchIndex + batchSize_ <= totalNumData:
            newBatchIndex = self.globalBatchIndex + batchSize_
            batchIndices = list(range(self.globalBatchIndex, newBatchIndex))

        else:
            newBatchIndex = self.globalBatchIndex + batchSize_ - totalNumData
            batchIndices = list(range(self.globalBatchIndex, totalNumData)) + list(range(0, newBatchIndex))

        # randomize within a batch (does this actually make a difference...? Don't think so.)
        np.random.shuffle(batchIndices)

        XBatch = self.XData_train[batchIndices]
        YBatch = self.YData_train[batchIndices]

        self.globalBatchIndex = newBatchIndex

        if verbose_:
            print('Indices:', batchIndices, '--> # tokens:', [len(d) for d in XBatch], '--> Y values:', YBatch)

        return XBatch, YBatch

    def get_all_training_data(self):
        x, y = self.XData_train, self.YData_train
        assert len(x)==len(y)

        return x, y

    def get_validation_data(self):
        x, y = self.XData_valid, self.YData_valid
        assert len(x) == len(y)

        return x, y

    def get_test_data(self):
        x, y = self.XData_test, self.YData_test
        assert len(x) == len(y)

        return x, y

    def get_classes(self):
        return set(self.YData)


if __name__ == '__main__':
    dataReader = DataReader(embeddingsFilename='data/glove/glove.6B/glove.6B.50d.txt')

    x, y = dataReader.get_next_training_batch(1)
    dataReader.get_all_training_data()
    dataReader.get_validation_data()
    dataReader.get_test_data()