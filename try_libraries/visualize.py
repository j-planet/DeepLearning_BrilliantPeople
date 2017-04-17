import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, stack_bidirectional_dynamic_rnn, MultiRNNCell, DropoutWrapper, LSTMCell
import matplotlib.pyplot as plt
from utilities import tensorflowFilewriters


VARIABLE_LENGTHS = True
USE_RELU = False    # relu or tanh

vecDim = 99
numClasses = 2

numSteps = 100
numSeq = 8
outputKeepProb = 1.

batchSize_training = 100
batchSize_testing = 100
testEvery = 10


def generate_batch(batchSize_):
    """
    :return: x, y, lengths 
    """

    res_x = np.random.random((batchSize_, numSeq, vecDim))

    if VARIABLE_LENGTHS:
        res_lengths = [np.random.randint(numSeq)+1 for _ in range(batchSize_)]
        # if above average length, class = 1, else 0. (i.e. depends solely on the lengths)
        # res_y = np.array([ [0, 1] if l > int(numSeq/2) else [1, 0] for l in res_lengths])

        # if sum of the last row is above average
        res_y = np.array([[0, 1] if res_x[i, l-1, :].sum() > vecDim*0.5 else [1, 0] for i, l in enumerate(res_lengths)])
    else:
        res_lengths = None
        res_y = np.array([[0, 1] if res_x[i, -1, :].sum() > vecDim*0.5 else [1, 0] for i in range(batchSize_)])

    return res_x, res_y, res_lengths

def last_relevant(output_, lengths_):

    if VARIABLE_LENGTHS:

        batch_size = tf.shape(output_)[0]
        max_length = tf.shape(output_)[1]
        out_size = int(output_.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (lengths_ - 1)
        flat = tf.reshape(output_, [-1, out_size])

        return tf.gather(flat, index)
    else:

        return output_[:,-1,:]


def make_stacked_cells(numLayers_, numCellUnits_):

    def _make_base_cell():
        return LSTMCell(numCellUnits_, activation=tf.nn.relu if USE_RELU else tf.tanh)

    if outputKeepProb == 1.0:
        return [_make_base_cell() for _ in range(numLayers_)]

    return [DropoutWrapper(_make_base_cell(), output_keep_prob=outputKeepProb) for _ in range(numLayers_)]

def variable_summaries(var, nameScope):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(nameScope):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def get_variable_by_name(name_):
    l = [v for v in tf.global_variables() if v.name == name_]

    return l[0] if len(l)>0 else None

def get_variable_by_name_regex(regexStr_):
    return [v for v in tf.global_variables() if len(re.findall(regexStr_, v.name)) > 0]

def add_1_dimension(tensor_):
    return tf.stack([tensor_], axis=-1)

def main(useCorrectLengths_, useStackRNN_, numLayers_, numCellUnits_, learningRate_, outputDir_):

    if outputDir_:
        print('LOGGED TO', outputDir_)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    title = '%s (%s lengths, %d units, %d layers, %0.3f lr)' % \
            ('stacked' if useStackRNN_ else 'notstacked',
             'CORRECT' if useCorrectLengths_ else 'WRONG',
             numCellUnits_, numLayers_, learningRate_)

    print('=============', title, '=============')

    x = tf.placeholder(tf.float32, [None, None, vecDim], name='x')
    y = tf.placeholder(tf.float32, [None, numClasses], name='y')
    sequenceLength = tf.placeholder(tf.int32, name='sequenceLength') if VARIABLE_LENGTHS else None

    with tf.name_scope('bidirLayer'):
        with tf.name_scope('forwardLSTMs'):
            forwardCells = make_stacked_cells(numLayers_, numCellUnits_)

        with tf.name_scope('backwardLSTMs'):
            backwardCells = make_stacked_cells(numLayers_, numCellUnits_)

        if useStackRNN_:
            outputs, _, _ = stack_bidirectional_dynamic_rnn(forwardCells, backwardCells,
                                                            inputs=x, dtype=tf.float32,
                                                            sequence_length=sequenceLength)
        else:
            outputs, states = tf.nn.bidirectional_dynamic_rnn(MultiRNNCell(forwardCells), MultiRNNCell(backwardCells),
                                                              time_major=False, inputs=x, dtype=tf.float32,
                                                              sequence_length=sequenceLength,
                                                              swap_memory=True)
            outputs = tf.concat(outputs, 2)

        output = last_relevant(outputs, sequenceLength)
        variable_summaries(output, 'output')

    weights = tf.Variable(tf.random_normal([2 * numCellUnits, numClasses]), name='weights')
    variable_summaries(weights, 'weights')

    biases = tf.Variable(tf.random_normal([numClasses]), name='biases')
    variable_summaries(biases, 'biases')

    with tf.name_scope('evaluation'):
        logits = tf.matmul(output, weights) + biases
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate_).minimize(cost)

    # predictions and accuracy
    pred = tf.argmax(logits, 1)
    trueY = tf.argmax(y, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, trueY), tf.float32))

    tf.summary.scalar('cost', cost)
    tf.summary.scalar('accuracy', accuracy)

    if outputDir_:
        mergedSummaries = tf.summary.merge_all()
        trainWriter, testWriter = tensorflowFilewriters(outputDir_)
        trainWriter.add_graph(sess.graph)

    # only for tesitng/validation

    # =========== TRAIN!! ===========
    sess.run(tf.global_variables_initializer())
    testX, testY, testLengths = generate_batch(batchSize_testing)
    trainAccVec = []
    trainCostVec = []
    testXPts = []
    testAccVec = []
    testCostVec = []

    for step in range(numSteps):
        numDataPoints = (step+1) * batchSize_training

        batchX, batchY, xLengths = generate_batch(batchSize_training)
        feedDict = {x: batchX, y: batchY,
                    sequenceLength: xLengths if useCorrectLengths_ else [numSeq] * batchSize_training} if VARIABLE_LENGTHS \
            else {x: batchX, y: batchY}



        # what happens if we use the wrong lengths
        _, c, acc, summaryRes = sess.run([optimizer, cost, accuracy, mergedSummaries], feedDict)
        trainCostVec.append(c)
        trainAccVec.append(acc)

        print('Step %d (%d data points): training cost = %0.3f accuracy = %0.3f'
              % (step, numDataPoints, c, acc))

        if outputDir_:
            trainWriter.add_summary(summaryRes, numDataPoints)

        if step % testEvery == 0:
            feedDict = {x: testX, y: testY,
                        sequenceLength: testLengths if useCorrectLengths_ else [numSeq] * batchSize_testing} if VARIABLE_LENGTHS \
                else {x: testX, y: testY}

            images = [
                tf.summary.image('outputs'+str(numDataPoints), add_1_dimension(tf.transpose(outputs, [1, 0, 2])), max_outputs=numSeq),
                tf.summary.image('fw_weights_cell_0_Adam'+str(numDataPoints),
                                 add_1_dimension(tf.stack(get_variable_by_name_regex('fw.*cell_0.*weights.*Adam'))),
                                 max_outputs=2),
                tf.summary.image('bw_weights_cell_0_Adam'+str(numDataPoints),
                                 add_1_dimension(tf.stack(get_variable_by_name_regex('bw.*cell_0.*weights.*Adam'))),
                                 max_outputs=2),
                tf.summary.image('fw_biases_Adam'+str(numDataPoints), add_1_dimension(
                    tf.reshape(tf.stack(get_variable_by_name_regex('fw.*cell.*biases.*Adam')),
                               (2 * numLayers_, -1, 1))), max_outputs=2 * numLayers_),
                tf.summary.image('bw_biases_Adam'+str(numDataPoints), add_1_dimension(
                    tf.reshape(tf.stack(get_variable_by_name_regex('bw.*cell.*biases.*Adam')),
                               (2 * numLayers_, -1, 1))), max_outputs=2 * numLayers_)]
            if numLayers_>1:
                images += [tf.summary.image('fw_weights_cell_1on_Adam'+str(numDataPoints),
                                            add_1_dimension(tf.stack(get_variable_by_name_regex('fw.*cell_[1-9].*weights.*Adam'))),
                                            max_outputs=2 * (numLayers_ - 1)),
                           tf.summary.image('bw_weights_cell_1on_Adam'+str(numDataPoints),
                                            add_1_dimension(tf.stack(get_variable_by_name_regex('bw.*cell_[1-9].*weights.*Adam'))),
                                            max_outputs=2 * (numLayers_ - 1))]

            testC, testAcc, summaryRes, imgs = sess.run([cost, accuracy, mergedSummaries, images], feedDict)

            testXPts.append(numDataPoints)
            testCostVec.append(testC)
            testAccVec.append(testAcc)
            print('>>>> test cost = %0.3f accuracy = %0.3f' % (testC, testAcc))

            if outputDir_:
                testWriter.add_summary(summaryRes, numDataPoints)
                for img in imgs:
                    testWriter.add_summary(img, numDataPoints)


    xPts = (np.arange(numSteps)+1)*batchSize_training

    figure = plt.figure()
    figure.suptitle(title)

    ax1 = figure.add_subplot(221)
    ax1.set_title('Training Costs')
    ax1.set_xlabel('# data points')
    ax1.set_ylim(0, 1)
    ax1.plot(xPts, trainCostVec)

    ax2 = figure.add_subplot(222)
    ax2.set_title('Training Accuracies')
    ax2.set_xlabel('# data points')
    ax2.set_ylim(0, 1)
    ax2.plot(xPts, trainAccVec)


    ax3 = figure.add_subplot(223)
    ax3.set_title('Testing Costs')
    ax3.set_xlabel('# data points')
    ax3.set_ylim(0, 1)
    ax3.plot(testXPts, testCostVec)

    ax4 = figure.add_subplot(224)
    ax4.set_title('Testing Accuracies')
    ax4.set_xlabel('# data points')
    ax4.set_ylim(0, 1)
    ax4.plot(testXPts, testAccVec)

    if outputDir_ is not None:
        figure.savefig(os.path.join(outputDir_, title + '.png'))

if __name__ == '__main__':

    rootOutputDir = os.path.join('../logs/', 'testSummaries')

    for useCorrectLengths in [True]:
        for useStackRNN in [False]:
            for numLayers in [1]:
                for numCellUnits in [101]:
                    for learningRate in [0.001]:

                        outputDir = os.path.join(rootOutputDir,
                                                 '%dlayers_%dunits_%s_%s' % (numLayers, numCellUnits, 'relu' if USE_RELU else 'tanh', 'varLen' if VARIABLE_LENGTHS else 'fixedLen'))
                        if not os.path.exists(outputDir): os.mkdir(outputDir)

                        main(useCorrectLengths_ = useCorrectLengths,
                             useStackRNN_ = useStackRNN,
                             numLayers_ = numLayers, numCellUnits_=numCellUnits, learningRate_=learningRate,
                             outputDir_=outputDir)


    plt.show()
