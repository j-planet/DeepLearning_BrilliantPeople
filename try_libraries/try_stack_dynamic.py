import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, stack_bidirectional_dynamic_rnn, MultiRNNCell
import matplotlib.pyplot as plt



vecDim = 100
numClasses = 2

numCellUnits = 5
numSteps = 300
numSeq = 8

batchSize_training = 25
batchSize_testing = 300
testEvery = 10


def generate_batch(batchSize_):
    """
    :return: x, y, lengths 
    """

    res_x = np.random.random((batchSize_, numSeq, vecDim))
    res_lengths = [np.random.randint(numSeq)+1 for _ in range(batchSize_)]

    # if above average length, class = 1, else 0. (i.e. depends solely on the lengths)
    # res_y = np.array([ [0, 1] if l > int(numSeq/2) else [1, 0] for l in res_lengths])

    # if sum of the last row is above average
    res_y = np.array([[0, 1] if res_x[i, l-1, :].sum() > vecDim*0.5 else [1, 0] for i, l in enumerate(res_lengths)])

    return res_x, res_y, res_lengths

def last_relevant(output_, lengths_):
    batch_size = tf.shape(output_)[0]
    max_length = tf.shape(output_)[1]
    out_size = int(output_.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (lengths_ - 1)
    flat = tf.reshape(output_, [-1, out_size])

    return tf.gather(flat, index)


def make_stacked_cells(numLayers_):
    return [BasicLSTMCell(numCellUnits) for _ in range(numLayers_)]


def main(useCorrectLengths_, useStackRNN_, numLayers_, learningRate_, outputDir_):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    title = '%s (%s lengths, %d layers, %0.3f learning rate)' % \
            ('stack_bidirectional_dynamic_rnn' if useStackRNN_ else 'bidirectional_dynamic_rnn',
             'CORRECT' if useCorrectLengths_ else 'WRONG',
             numLayers_, learningRate_)

    print('=============', title, '=============')

    x = tf.placeholder(tf.float32, [None, None, vecDim])
    y = tf.placeholder(tf.float32, [None, numClasses])
    sequenceLength = tf.placeholder(tf.int32)

    weights = tf.Variable(tf.random_normal([2 * numCellUnits, numClasses]), name='weights')
    biases = tf.Variable(tf.random_normal([numClasses]), name='biases')

    forwardCells = make_stacked_cells(numLayers_)
    backwardCells = make_stacked_cells(numLayers_)

    # outputs = tf.concat(
    #     tf.nn.bidirectional_dynamic_rnn(MultiRNNCell(forwardCells), MultiRNNCell(backwardCells),
    #                                     time_major=False, inputs=x, dtype=tf.float32,
    #                                     sequence_length=sequenceLength,
    #                                     swap_memory=True)[0],
    #     2)

    outputs = \
        stack_bidirectional_dynamic_rnn(forwardCells, backwardCells,
                                        inputs=x, dtype=tf.float32,
                                        sequence_length=sequenceLength)[0] if useStackRNN_ \
            else tf.concat(
            tf.nn.bidirectional_dynamic_rnn(MultiRNNCell(forwardCells), MultiRNNCell(backwardCells),
                                            time_major=False, inputs=x, dtype=tf.float32,
                                            sequence_length=sequenceLength,
                                            swap_memory=True)[0],
            2)

    output = last_relevant(outputs, sequenceLength)
    logits = tf.matmul(output, weights) + biases
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate_).minimize(cost)

    # predictions and accuracy
    pred = tf.argmax(logits, 1)
    trueY = tf.argmax(y, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, trueY), tf.float32))

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

        # what happens if we use the wrong lengths
        _, c, acc = sess.run([optimizer, cost, accuracy],
                             feed_dict={x: batchX, y: batchY,
                                        sequenceLength: xLengths if useCorrectLengths_ else [numSeq] * batchSize_training})
        trainCostVec.append(c)
        trainAccVec.append(acc)

        print('Step %d (%d data points): training cost = %0.3f accuracy = %0.3f'
              % (step, numDataPoints, c, acc))

        if step % testEvery == 0:
            testC, testAcc = sess.run([cost, accuracy],
                                      feed_dict={x: testX, y: testY, sequenceLength: testLengths if useCorrectLengths_ else [numSeq] * batchSize_testing})

            testXPts.append(numDataPoints)
            testCostVec.append(testC)
            testAccVec.append(testAcc)
            print('>>>> test cost = %0.3f accuracy = %0.3f' % (testC, testAcc))


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

    outputDir = os.path.join('../logs/', 'testStackRnn')
    if not os.path.exists(outputDir): os.mkdir(outputDir)

    for useCorrectLengths in [True]:
        for useStackRNN in [True, False]:
            for numLayers in [1, 2, 10]:
                for learningRate in [0.05, 0.001]:

                    main(useCorrectLengths_ = useCorrectLengths,
                         useStackRNN_ = useStackRNN,
                         numLayers_ = numLayers, learningRate_=learningRate,
                         outputDir_=outputDir)

    # main(useCorrectLengths_=True,
    #      useStackRNN_=False,
    #      numLayers_=10,
    #      outputDir_=None)

    # plt.show()
