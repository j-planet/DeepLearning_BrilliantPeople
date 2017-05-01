import tensorflow as tf
import json, os, glob
import numpy as np

from data_readers.abstract_data_reader import AbstractDataReader
from data_processing.file2vec import filename2name


PPL_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/peopleData/')

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


class EmbeddingDataReader(AbstractDataReader):

    def __init__(self, inputFilesDir, bucketingOrRandom, batchSize_, minimumWords,
                 loggerFactory=None, train_valid_test_split_=(0.8, 0.1, 0.1), padToFull=False):

        self.padToFull = padToFull

        super().__init__(inputFilesDir, bucketingOrRandom, batchSize_, minimumWords,
                         loggerFactory, train_valid_test_split_)

        self.print('padToFull: ' + str(padToFull))

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
                numSkipped += 1
                continue

            XData.append(mat)
            occ = d['occupation']
            xLengths.append(mat.shape[0])
            YData.append(occ if type(occ) == str else occ[-1])
            names.append(filename2name(inputFilename))

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
            x = patch_arrays(xData_[start:stop], xLengths_[start:stop], self.maxXLen if self.padToFull else None)
            res.append((x, yData_[start:stop], xLengths_[start:stop], names_[start:stop]))

        return res

    @classmethod
    def premade_sources(cls):

        def _p(d):
            return os.path.abspath(os.path.join(PPL_DATA_DIR, d))

        return {'tiny_fake_2': _p('2_samples'),
                'tiny_fake_4': _p('4_samples'),
                'small_2occupations': _p('earlyLifesWordMats/politician_scientist'),
                'small': _p('earlyLifesWordMats'),
                'full_2occupations': _p('earlyLifesWordMats_42B300d/politician_scientist'),
                'full': _p('earlyLifesWordMats_42B300d')}


if __name__ == '__main__':
    dr = EmbeddingDataReader(os.path.join(PPL_DATA_DIR, '2_samples'), 'bucketing', 5, 1, padToFull=False)

    for _ in range(10):
        d, names = dr.get_next_training_batch()

        print(d[dr.input['x']].shape)
        print(names)
