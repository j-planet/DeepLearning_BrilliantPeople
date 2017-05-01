import tensorflow as tf
from tensorflow.contrib.learn import preprocessing
import json, os
import numpy as np

from data_readers.abstract_data_reader import AbstractDataReader


PPL_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/peopleData/')


class TextDataReader(AbstractDataReader):

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

                if numTokens < self.minimumWords:
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

    @classmethod
    def premade_sources(cls):

        def _p(d): return os.path.join(PPL_DATA_DIR, 'tokensfiles', d)

        return {'tiny_fake_2': _p('fake2.json'),
                'full_2occupations': _p('pol_sci.json'),
                'full': _p('all.json')}



if __name__ == '__main__':

    # dataReader = TextDataReader('./data/peopleData/tokensfiles/all.json', 'bucketing', 5, 1)
    dataReader = TextDataReader.maker_from_premade_source('full_2occupations')(bucketingOrRandom='bucketing', batchSize_=5, minimumWords=1)

    for _ in range(10):
        d, names = dataReader.get_next_training_batch()

        print(d)
        print(names)
