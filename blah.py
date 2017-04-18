from pprint import pprint
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import summary
from tensorflow.python.client.timeline import Timeline
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, static_bidirectional_rnn, stack_bidirectional_dynamic_rnn, MultiRNNCell, DropoutWrapper, LSTMCell


from data_reader import DataReader
from utilities import tensorflowFilewriters, reshape_x_for_non_dynamic


PATCH_TO_FULL = False
LOG_DIR = './logs/main'
saveImages = False

# ================== DATA ===================
with tf.device('/cpu:0'):
    # dataReader = DataReader('./data/peopleData/2_samples', 'bucketing')
    dataReader = DataReader('./data/peopleData/earlyLifesWordMats_42B300d/politician_scientist', 'bucketing')
    # dataReader = DataReader('./data/peopleData/earlyLifesWordMats')
    # dataReader = DataReader('./data/peopleData/earlyLifesWordMats_42B300d', 'bucketing')

# sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)))
sess = tf.InteractiveSession()

# ================== CONFIG ===================

# --------- network ---------
vecDim = 300
numHiddenLayerFeatures = 32
numRnnLayers = 10
outputKeepProbConstant = 0.99
usePeepholes = True

numClasses = len(dataReader.get_classes_labels())
outputKeepProb = tf.placeholder(tf.float32)

# --------- running ---------
learningRateConstant = 0.01
numSteps = 100  # 1 step runs 1 batch
batchSize = 20

logTrainingEvery = 2
logValidationEvery = 5

# --------- constant 'variables' ---------
learningRate = tf.Variable(learningRateConstant, name='learningRate')
validCost = tf.Variable(1, name='validationCost')
validAcc = tf.Variable(0, name='validationAccuracy')
summary.scalar('valid cost', validCost)
summary.scalar('valid accuracy', validAcc)


def validate_or_test(batchSize_, validateOrTest):

    assert validateOrTest in ['validate', 'test']

    data = dataReader.get_data_in_batches(batchSize_, validateOrTest, patchTofull_=PATCH_TO_FULL)

    totalCost = 0
    totalAccuracy = 0
    allTrueYInds = []
    allPredYInds = []
    allNames = []

    # d: x, y, xLengths, names
    for d in data:

        feedDict = {x: d[0], y: d[1], sequenceLength: d[2], outputKeepProb: outputKeepProbConstant}
        c, acc, trueYInds, predYInds = sess.run([cost, accuracy, trueY, pred], feed_dict=feedDict)

        actualCount = len(d[2])
        totalCost += c * actualCount
        totalAccuracy += acc * actualCount
        allNames += list(d[3])
        allTrueYInds += list(trueYInds)
        allPredYInds += list(predYInds)

    assert len(allTrueYInds)==len(allPredYInds)==len(allNames)

    numDataPoints = len(allTrueYInds)
    avgCost = totalCost / numDataPoints
    avgAccuracy = totalAccuracy / numDataPoints

    if validateOrTest=='validate':
        sess.run(tf.assign(validCost, avgCost))
        sess.run(tf.assign(validAcc, avgAccuracy))

    labels = dataReader.get_classes_labels()
    print('loss = %.3f, accuracy = %.3f' % (avgCost, avgAccuracy))
    print('True label became... --> ?')
    for i, name in enumerate(allNames):
        print('%s: %s --> %s %s' %
              (name,
               labels[allTrueYInds[i]], labels[allPredYInds[i]],
               '(wrong)' if allTrueYInds[i] != allPredYInds[i] else ''))

def print_log_str(x_, y_, xLengths_, names_):
    """
    :return a string of loss and accuracy
    """

    feedDict = {x: x_, y: y_, sequenceLength: xLengths_, outputKeepProb: outputKeepProbConstant}

    labels = dataReader.get_classes_labels()
    c, acc, trueYInds, predYInds = sess.run([cost, accuracy, trueY, pred], feed_dict=feedDict)

    print('loss = %.3f, accuracy = %.3f' % (c, acc))
    print('True label became... --> ?')
    for i, name in enumerate(names_):
        print('%s: %s --> %s %s' %
              (name,
               labels[trueYInds[i]], labels[predYInds[i]],
               '(wrong)' if trueYInds[i]!=predYInds[i] else '' ))

def last_relevant(output_, lengths_):
    batch_size = tf.shape(output_)[0]
    max_length = tf.shape(output_)[1]
    out_size = int(output_.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (lengths_ - 1)
    flat = tf.reshape(output_, [-1, out_size])

    return tf.gather(flat, index)

def save_matrix_img(mats_, title, outputDir_, transpose_=False):

    d = np.array(mats_) if len(mats_[0].shape) == 1 else np.concatenate(mats_, axis=1)

    fig = plt.figure()
    ax = plt.subplot(111)
    heatmap = ax.matshow(np.transpose(d) if transpose_ else d, cmap='gray')
    plt.colorbar(heatmap)
    plt.title(title)
    fig.savefig(os.path.join(outputDir_, title+'.png'))


if __name__ == '__main__':
    st = time()

    print('====== CONFIG: SHUFFLED %d hidden layer(s) with %d features each; '
          'dropoutKeep = %0.2f'
          ' batch size %d, initial learning rate %.3f'
          % (numRnnLayers, numHiddenLayerFeatures, outputKeepProbConstant, batchSize, learningRateConstant))
    print('usePeepholes', usePeepholes)

    # ================== GRAPH ===================
    x = tf.placeholder(tf.float32, [None, None, vecDim])
    # x = tf.placeholder(tf.float32, [None, dataReader.get_max_len(), vecDim])
    y = tf.placeholder(tf.float32, [None, numClasses])
    sequenceLength = tf.placeholder(tf.int32)

    # weights = tf.Variable(tf.random_normal([numHiddenLayerFeatures, numClasses]))
    weights = tf.Variable(tf.random_normal([2*numHiddenLayerFeatures, numClasses]), name='weights')
    biases = tf.Variable(tf.random_normal([numClasses]), name='biases')


    def make_BasicLSTMCell():
        return BasicLSTMCell(numHiddenLayerFeatures)

    def make_BasicRNNCell():
        return BasicRNNCell(numHiddenLayerFeatures)

    def make_stacked_cells(baseCellMaker_):

        if outputKeepProb == 1.:    # no drop out
            return [baseCellMaker_() for _ in range(numRnnLayers)]

        return [DropoutWrapper(baseCellMaker_(), output_keep_prob=outputKeepProb) for _ in range(numRnnLayers)]


    forwardCells = make_stacked_cells(make_BasicLSTMCell)
    backwardCells = make_stacked_cells(make_BasicLSTMCell)


    outputs, _, _ \
        = stack_bidirectional_dynamic_rnn(forwardCells, backwardCells,
                                          inputs=x, dtype=tf.float32,
                                          sequence_length=sequenceLength)


    # outputs = tf.concat(
    #     tf.nn.bidirectional_dynamic_rnn(forwardCells, backwardCells,
    #                                     time_major=False, inputs=x, dtype=tf.float32,
    #                                     sequence_length=sequenceLength,
    #                                     swap_memory=True)[0], 2)


    # cost and optimize
    output = last_relevant(outputs, sequenceLength)
    logits = tf.matmul(output, weights) + biases
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRateConstant).minimize(cost)

    # predictions and accuracy
    pred = tf.argmax(logits, 1)
    trueY = tf.argmax(y, 1)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(pred, trueY)
        , tf.float32))

    summary.scalar('training cost', cost)
    summary.scalar('training accuracy', accuracy)

    # =========== set up tensorboard ===========
    merged_summaries = summary.merge_all()
    train_writer, valid_writer = tensorflowFilewriters(LOG_DIR)
    train_writer.add_graph(sess.graph)

    # =========== TRAIN!! ===========
    sess.run(tf.global_variables_initializer())
    dataReader.start_batch_from_beginning()     # technically unnecessary

    # run_metadata = tf.RunMetadata()
    # nrows = int(numSteps ** 0.5)
    # ncols = int(np.ceil(numSteps / nrows))

    for step in range(numSteps):
        numDataPoints = (step+1) * batchSize
        print('\nStep %d (%d data points); learning rate = %0.3f:' % (step, numDataPoints, sess.run(learningRate)))

        lrDecay = 0.95 ** (numDataPoints / len(dataReader.train_indices))
        sess.run(tf.assign(learningRate, max(learningRateConstant * lrDecay, 1e-4)))

        batchX, batchY, xLengths, names = dataReader.get_next_training_batch(batchSize, patchTofull_=PATCH_TO_FULL, verbose_=False)
        feedDict = {x: batchX, y: batchY, sequenceLength: xLengths, outputKeepProb: outputKeepProbConstant}

        _, summaries, w, b, lstmVars, outputNums = \
            sess.run([optimizer, merged_summaries, weights, biases, tf.trainable_variables()[5:], output]
                     , feed_dict=feedDict)

        # options=tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE),
        # run_metadata=run_metadata)
        # trace = Timeline(step_stats=run_metadata.step_stats)

        # print evaluations
        if step % logTrainingEvery == 0:
            train_writer.add_summary(summaries, step * batchSize)
            print_log_str(batchX, batchY, xLengths, names)
            train_writer.flush()

        if step % logValidationEvery == 0:
            # valid_writer.add_summary(summaries, step * batchSize)
            print('\n>>> Validation:')
            validate_or_test(10, 'validate')

    print('Time elapsed:', time()-st)

    print('\n>>>>>> Test:')
    validate_or_test(10, 'test')


    # trace_file = open('timeline.ctf.json', 'w')
    # trace_file.write(trace.generate_chrome_trace_format())
