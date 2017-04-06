from pprint import pprint
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

from data_reader import DataReader
from utilities import tensorflowFilewriter


def print_log_str(x_, y_, xLengths_):
    """
    :return a string of loss and accuracy
    """

    feedDict = {x: x_, y: y_, sequenceLength: xLengths_}

    print('loss = %.3f, accuracy = %.3f' % \
          tuple(sess.run([cost, accuracy], feed_dict=feedDict)))

    print('True label became... --> ?')
    pprint(['%s --> %s' % (dataReader.get_classes_labels()[t], dataReader.get_classes_labels()[p]) for t, p in
            zip(*sess.run([trueY, pred], feed_dict=feedDict))])



sess = tf.InteractiveSession()
logger_train = tensorflowFilewriter('./logs/train')
logger_train.add_graph(sess.graph)

# ================== DATA ===================
dataReader = DataReader(vectorFilesDir='./data/peopleData/earlyLifesWordMats')

# ================== CONFIG ===================

# --------- network ---------
vecDim = 300
numHiddenLayerFeatures = 128
numClasses = len(dataReader.get_classes_labels())

# --------- running ---------
learningRate = 0.001
numSteps = 1000     # 1 step runs 1 batch
batchSize = 5

logTrainingEvery = 10
logValidationEvery = 100

print('====== CONFIG: batch size %d, learning rate %.3f' % (batchSize, learningRate))

# ================== GRAPH ===================
# numSequences = dataReader.get_max_len()
x = tf.placeholder('float', [None, None, vecDim])
y = tf.placeholder('float', [None, numClasses])
sequenceLength = tf.placeholder(tf.int32)

weights = tf.Variable(tf.random_normal([2*numHiddenLayerFeatures, numClasses]))
biases = tf.Variable(tf.random_normal([numClasses]))

# make LSTM cells
lstmCell_forward = BasicLSTMCell(numHiddenLayerFeatures)
lstmCell_backward = BasicLSTMCell(numHiddenLayerFeatures)

# wrap RNN around LSTM cells
outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstmCell_forward, lstmCell_backward,
                                             time_major=False, inputs=x, dtype=tf.float32,
                                             sequence_length=sequenceLength)

# cost and optimize
logits = tf.matmul(tf.concat(outputs, 2)[:,-1,:], weights) + biases
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# predictions and accuracy
pred = tf.argmax(logits, 1)
trueY = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(
    tf.equal(pred, trueY)
    , tf.float32))

# train
sess.run(tf.global_variables_initializer())
dataReader.start_batch_from_beginning()     # technically unnecessary

for step in range(numSteps):

    # will prob need some shape mangling here
    batchX, batchY, xLengths = dataReader.get_next_training_batch(batchSize, verbose_=False)
    feedDict = {x: batchX, y: batchY, sequenceLength: xLengths}

    sess.run(optimizer, feed_dict=feedDict)

    # print evaluations
    if step % logTrainingEvery == 0:
        print('\nStep %d (%d data points):' % (step, step*batchSize))
        print_log_str(batchX, batchY, xLengths)

        if step % logValidationEvery == 0:
            print('>>> Validation:')
            print_log_str(*(dataReader.get_validation_data()))



print('\n>>>>>> Test:', print_log_str(*(dataReader.get_test_data())))





