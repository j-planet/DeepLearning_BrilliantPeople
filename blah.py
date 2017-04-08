from pprint import pprint
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, static_bidirectional_rnn

from data_reader import DataReader
from utilities import tensorflowFilewriter


st = time()
sess = tf.InteractiveSession()

DEVICE = '/gpu:0'
PATCH_TO_FULL = False

# with tf.device(DEVICE):

def print_log_str(x_, y_, xLengths_):
    """
    :return a string of loss and accuracy
    """

    # feedDict = {x: x_, y: y_}
    feedDict = {x: x_, y: y_, sequenceLength: xLengths_}

    print('loss = %.3f, accuracy = %.3f' % \
          tuple(sess.run([cost, accuracy], feed_dict=feedDict)))

    print('True label became... --> ?')
    pprint(['%s --> %s' % (dataReader.get_classes_labels()[t], dataReader.get_classes_labels()[p]) for t, p in
            zip(*sess.run([trueY, pred], feed_dict=feedDict))])


logger_train = tensorflowFilewriter('./logs/train')
logger_train.add_graph(sess.graph)

# ================== DATA ===================
# dataReader = DataReader(vectorFilesDir='./data/peopleData/earlyLifesWordMats')
dataReader = DataReader(vectorFilesDir='./data/peopleData/earlyLifesWordMats/politician_scientist')

# ================== CONFIG ===================

# --------- network ---------
vecDim = 300
numHiddenLayerFeatures = 128
numClasses = len(dataReader.get_classes_labels())

# --------- running ---------
learningRate = 0.1
numSteps = 10     # 1 step runs 1 batch
batchSize = 10

logTrainingEvery = 1
logValidationEvery = 3

print('====== CONFIG: SHUFFLED batch size %d, learning rate %.3f' % (batchSize, learningRate))

# ================== GRAPH ===================
x = tf.placeholder(tf.float32, [None, None, vecDim])
# x = tf.placeholder(tf.float32, [None, dataReader.get_max_len(), vecDim])
y = tf.placeholder(tf.float32, [None, numClasses])
sequenceLength = tf.placeholder(tf.int32)

weights = tf.Variable(tf.random_normal([2*numHiddenLayerFeatures, numClasses]))
biases = tf.Variable(tf.random_normal([numClasses]))

# make LSTM cells
lstmCell_forward = BasicLSTMCell(numHiddenLayerFeatures)
lstmCell_backward = BasicLSTMCell(numHiddenLayerFeatures)

# wrap RNN around LSTM cells
outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstmCell_forward, lstmCell_backward,
                                             time_major=False, inputs=x, dtype=tf.float32,
                                             sequence_length=sequenceLength,
                                             swap_memory=True)

# outputs, _ = static_bidirectional_rnn(lstmCell_forward, lstmCell_backward,
#                                       inputs=x, dtype=tf.float32)

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

    print('\nStep %d (%d data points):' % (step, step * batchSize))

    # will prob need some shape mangling here
    batchX, batchY, xLengths = dataReader.get_next_training_batch(batchSize, patchTofull_=PATCH_TO_FULL, verbose_=False)
    # feedDict = {x: batchX, y: batchY}
    feedDict = {x: batchX, y: batchY, sequenceLength: xLengths}

    sess.run(optimizer, feed_dict=feedDict)

    # print evaluations
    if step % logTrainingEvery == 0:
        print_log_str(batchX, batchY, xLengths)

        if step % logValidationEvery == 0:
            print('>>> Validation:')
            print_log_str(*(dataReader.get_validation_data(patchTofull_=PATCH_TO_FULL)))


print('\n>>>>>> Test:')
print_log_str(*(dataReader.get_test_data(patchTofull_=PATCH_TO_FULL)))

print('Time elapsed:', time()-st)


