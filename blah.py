from pprint import pprint
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import summary
from tensorflow.python.client.timeline import Timeline
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, static_bidirectional_rnn, MultiRNNCell, DropoutWrapper, LSTMCell

from data_reader import DataReader
from utilities import tensorflowFilewriters


PATCH_TO_FULL = False
LOG_DIR = './logs/main'

# ================== DATA ===================
with tf.device('/cpu:0'):
    # dataReader = DataReader('./data/peopleData/2_samples', 'bucketing')
    dataReader = DataReader('./data/peopleData/earlyLifesWordMats/politician_scientist', 'bucketing')
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
numSteps = 50  # 1 step runs 1 batch
batchSize = 25

logTrainingEvery = 5
logValidationEvery = 20

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

    print('====== CONFIG: SHUFFLED %d hidden layers with %d features each; '
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

    # make LSTM cells
    # singleLSTMCell = LSTMCell(numHiddenLayerFeatures, use_peepholes=usePeepholes)
    singleLSTMCell = BasicLSTMCell(numHiddenLayerFeatures)

    cellsForward = [DropoutWrapper(singleLSTMCell, output_keep_prob=outputKeepProb)] * numRnnLayers
    cellsBackward = [DropoutWrapper(singleLSTMCell, output_keep_prob=outputKeepProb)] * numRnnLayers

    stackedCellsForward = MultiRNNCell(cellsForward)
    stackedCellsBackward = MultiRNNCell(cellsForward)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(stackedCellsForward, stackedCellsBackward,
                                                      time_major=False, inputs=x, dtype=tf.float32,
                                                      sequence_length=sequenceLength,
                                                      swap_memory=True)

    # wrap RNN around LSTM cells
    # baseCell = BasicLSTMCell(numHiddenLayerFeatures)
    # baseCellWDropout = DropoutWrapper(baseCell, output_keep_prob=outputKeepProb)
    # multiCell = MultiRNNCell([baseCell]*numRnnLayers)
    # outputs, _ = tf.nn.dynamic_rnn(multiCell,
    #                                time_major=False, inputs=x, dtype=tf.float32,
    #                                sequence_length=sequenceLength,
    #                                swap_memory=True)

    # cost and optimize
    # output = tf.concat(outputs, 2)[:,-1,:]
    output = last_relevant(tf.concat(outputs, 2), sequenceLength)

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

    all_outputs = []
    all_weights = []
    all_biases = []
    # lstm_A_ws = []
    # lstm_A_bs = []
    # lstm_B_ws = []
    # lstm_B_bs = []
    lstm_fwd_weights = [[] for _ in range(numRnnLayers)]
    lstm_fwd_biases = [[] for _ in range(numRnnLayers)]
    lstm_back_weights = [[] for _ in range(numRnnLayers)]
    lstm_back_biases = [[] for _ in range(numRnnLayers)]

    for step in range(numSteps):
        numDataPoints = (step+1) * batchSize
        print('\nStep %d (%d data points); learning rate = %0.3f:' % (step, numDataPoints, sess.run(learningRate)))

        lrDecay = 0.9 ** (numDataPoints / len(dataReader.train_indices))
        sess.run(tf.assign(learningRate, max(learningRateConstant * lrDecay, 1e-4)))

        batchX, batchY, xLengths, names = dataReader.get_next_training_batch(batchSize, patchTofull_=PATCH_TO_FULL, verbose_=False)
        feedDict = {x: batchX, y: batchY, sequenceLength: xLengths, outputKeepProb: outputKeepProbConstant}

        _, summaries, w, b, lstmVars, outputNums = \
            sess.run([optimizer, merged_summaries, weights, biases, tf.trainable_variables()[5:], output]
                     , feed_dict=feedDict)
        # options=tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE),
        # run_metadata=run_metadata)

        # print('here')
        # trace = Timeline(step_stats=run_metadata.step_stats)
        # print('done with here')


        # print evaluations
        if step % logTrainingEvery == 0:
            train_writer.add_summary(summaries, step * batchSize)
            print_log_str(batchX, batchY, xLengths, names)
            train_writer.flush()

            # pprint(w)
            # ax = plt.subplot(nrows, ncols, step+1)
            # heatmap = ax.matshow(w, cmap = 'gray')
            # plt.colorbar(heatmap)
            # plt.title('step %d, %d pts' % (step, numDataPoints))

            all_weights.append(w)
            all_weights.append(np.zeros((w.shape[0], 2)))

            all_biases.append(b)

            for i in range(numRnnLayers):
                lw_fwd = lstmVars[2*i]
                lb_fwd = lstmVars[2*i+1]

                lstm_fwd_weights[i].append(lw_fwd)
                lstm_fwd_weights[i].append(np.ones((lw_fwd.shape[0], 5)) * -0.5)
                lstm_fwd_biases[i].append(lb_fwd)

                lw_back = lstmVars[2 * i + 2*numRnnLayers]
                lb_back = lstmVars[2 * i + 1 + 2*numRnnLayers]

                lstm_back_weights[i].append(lw_back)
                lstm_back_weights[i].append(np.ones((lw_back.shape[0], 5)) * -0.5)
                lstm_back_biases[i].append(lb_back)

            # Aw, Ab, Bw, Bb = lstmVars
            # lstm_A_ws.append(Aw)
            # lstm_A_ws.append(np.ones((Aw.shape[0], 5)) * -0.5)
            #
            # lstm_A_bs.append(Ab)
            #
            # lstm_B_ws.append(Bw)
            # lstm_B_ws.append(np.ones((Bw.shape[0], 5)) * -0.5)
            #
            # lstm_B_bs.append(Bb)

            all_outputs.append(outputNums)
            all_outputs.append(np.ones((outputNums.shape[0], 3)) * -1)

        if step % logValidationEvery == 0:
            # valid_writer.add_summary(summaries, step * batchSize)
            print('\n>>> Validation:')
            validate_or_test(10, 'validate')


    print('Time elapsed:', time()-st)

    print('\n>>>>>> Test:')
    validate_or_test(10, 'test')

    plotsOutputDir = os.path.join(LOG_DIR, '%dlayers' % numRnnLayers)
    if not os.path.exists(plotsOutputDir): os.mkdir(plotsOutputDir)

    save_matrix_img(all_weights, 'w', plotsOutputDir)
    save_matrix_img(all_biases, 'b', plotsOutputDir)
    save_matrix_img(all_outputs, 'all_outputs', plotsOutputDir)

    for i in range(numRnnLayers):
        save_matrix_img(lstm_fwd_weights[i], 'lstm forward weights %d' % i, plotsOutputDir)
        save_matrix_img(lstm_fwd_biases[i], 'lstm forward biases %d' % i, plotsOutputDir, True)
        save_matrix_img(lstm_back_weights[i], 'lstm backward weights %d' % i, plotsOutputDir)
        save_matrix_img(lstm_back_biases[i], 'lstm backward biases %d' % i, plotsOutputDir, True)


    # trace_file = open('timeline.ctf.json', 'w')
    # trace_file.write(trace.generate_chrome_trace_format())
