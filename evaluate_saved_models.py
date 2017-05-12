import tensorflow as tf
from multiprocessing import cpu_count
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report, precision_recall_fscore_support

import matplotlib.pyplot as plt
import numpy as np
import itertools

from train import RunConfig
from data_readers.embedding_data_reader import EmbeddingDataReader
from models.mark6 import Mark6, RNNConfig


def evaluate_mark6(l2Scheme, rnnConfigs, pooledKeepProb, outputFpath = None):

    runConfig = RunConfig('medium')

    # make data reader
    dataReaderMaker = EmbeddingDataReader.maker_from_premade_source('full')
    dataReader = dataReaderMaker(bucketingOrRandom='bucketing', batchSize_=runConfig.batchSize, minimumWords=40)

    # make model
    p = dict([('initialLearningRate', 1e-3),
              ('l2RegLambda', 1e-6),
              ('l2Scheme', l2Scheme),

              # ('rnnConfigs', [RNNConfig([1024, 512], [0.6, 0.7])]),
              ('rnnConfigs', rnnConfigs),

              ('pooledKeepProb', pooledKeepProb),
              ('pooledActivation', None)
              ])

    modelMaker = lambda input_, logFac: Mark6(input_=input_, **p, loggerFactory_=logFac)
    model = modelMaker(dataReader.input, None)

    # make session
    numCores = cpu_count() - 1
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=numCores, inter_op_parallelism_threads=numCores)

    with tf.device('/cpu:0'):
        sess = tf.InteractiveSession(config=config)

    # restore from saved path
    savePath = '/Users/jj/Code/brilliant_people/logs/main/Mark6/loadmark6/saved/save.ckpt'
    tf.train.Saver().restore(sess, savePath)

    # feed in some data and evaluate
    res = [[ (model.evaluate(sess, fd, full=True), names) for
             fd, names in bg ]
           for bg in [dataReader.get_validation_data_in_batches(), dataReader.get_test_data_in_batches()]]

    if outputFpath is not None:
        with open(outputFpath, 'wb') as ofile:
            pickle.dump(res, ofile)

    return res


def plot_confusion_matrix(cm_, classLabels_, wrongOnly_):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    assert type(wrongOnly_) == bool

    cmap = plt.cm.Reds if wrongOnly_ else plt.cm.Blues
    title = 'Confusion Matrix' + (' (WRONG cases only)' if wrongOnly_ else '')

    if wrongOnly_:
        for i in range(cm_.shape[0]):
            cm_[i, i] = 0

    plt.figure()
    plt.imshow(cm_, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classLabels_))
    plt.xticks(tick_marks, classLabels_, rotation=45)
    plt.yticks(tick_marks, classLabels_)

    thresh = cm_.max() / 2.
    for i, j in itertools.product(range(cm_.shape[0]), range(cm_.shape[1])):
        plt.text(j, i, cm_[i, j],
                 horizontalalignment="center",
                 color="white" if cm_[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


fpath = 'metrics.bin'

# evaluate_mark6('final_stage',
#                [RNNConfig([64, 128, 256], [0.5, 0.6, 0.7]),
#                 RNNConfig([64, 64, 64, 64], [0.5, 0.6, 0.7, 0.8])],
#                1.,
#                None)

with open(fpath, 'rb') as ifile:
    data = pickle.load(ifile)

true_ys = []
pred_ys = []
names = []

for vt in data:
    for batch in vt:
        d, cur_names = batch

        names += list(cur_names)
        true_ys += list(d[2])
        pred_ys += list(d[3])

true_ys = np.array(true_ys)
pred_ys = np.array(pred_ys)
names = np.array(names)
labels = ['artist', 'athlete', 'author', 'businessman', 'entertainment', 'politician', 'scientist']

print('%-30s%12s%12s' % ('Name', 'True', 'Predicted'))
for t, p, n in zip(true_ys, pred_ys, names):
    if t!=p:
        print('%-30s%12s%12s' % (n, labels[t], labels[p]) )

prec, rec, fs, _ = precision_recall_fscore_support(true_ys, pred_ys)

for l, p, r, f in zip(labels, prec, rec, fs):
    print('%s | %0.3f | %0.3f | %0.3f' % (l, p, r, f))


cm = confusion_matrix(true_ys, pred_ys)
plot_confusion_matrix(cm, labels, wrongOnly_=False)
plot_confusion_matrix(cm, labels, wrongOnly_=True)

plt.show()
