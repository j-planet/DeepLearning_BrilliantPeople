import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from data_reader import DataReader
from utilities import tensorflowFilewriter


def log_str(x_, y_):
    """
    :return a string of loss and accuracy
    """

    return 'loss = %.3f, accuracy = %.3f' % \
           tuple(sess.run([cost, accuracy], feed_dict={x: x_, y: y_}))

sess = tf.InteractiveSession()
logger_train = tensorflowFilewriter('./logs/train')
logger_train.add_graph(sess.graph)

# ================== DATA ===================
dataReader = DataReader('./data/peopleData/embeddings/smallGlove.42B300d.txt')

# ================== CONFIG ===================

# --------- network ---------
stepSize = 50
numHiddenLayerFeatures = 128
numClasses = len(dataReader.get_classes_labels())

# --------- running ---------
learningRate = 0.001
numSteps = 1000     # 1 step runs 1 batch
batchSize = 10

logTrainingEvery = 10
logValidationEvery = 100

# ================== GRAPH ===================
x = tf.placeholder('float', [None, None, stepSize])
y = tf.placeholder('float', [None, numClasses])

weights = tf.Variable(tf.random_normal([2*numHiddenLayerFeatures, numClasses]))
biases = tf.Variable(tf.random_normal([numClasses]))

transformedX = tf.split(
    tf.reshape(
        tf.transpose(x, [1, 0, 2]),
        [-1, stepSize]),
    -1, 0
)

# make LSTM cells
lstmCell_forward = rnn.BasicLSTMCell(numHiddenLayerFeatures)
lstmCell_backward = rnn.BasicLSTMCell(numHiddenLayerFeatures)

# wrap RNN around LSTM cells
outputs, _, _ = rnn.static_bidirectional_rnn(lstmCell_forward, lstmCell_backward,
                                             transformedX, dtype=tf.float32)

# cost and optimize
logits = tf.matmul(outputs[-1], weights) + biases
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# predictions and accuracy
pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(
    tf.equal(pred, tf.argmax(y, 1))
    , tf.float32))

# train
sess.run(tf.global_variables_initializer())
dataReader.start_batch_from_beginning()     # technically unnecessary

for step in range(numSteps):

    # will prob need some shape mangling here
    batch_x, batch_y = dataReader.get_next_training_batch(batchSize, verbose_=False)

    sess.run(optimizer, {x: batch_x, y: batch_y})

    # print evaluations
    if step % logTrainingEvery == 0:
        print('\nStep %d (%d data points):' % (step, step*batchSize))
        print('Training:', log_str(batch_x, batch_y))

        if step % logValidationEvery == 0:
            print('>>> Validation:', log_str(*(dataReader.get_validation_data())))



print('\n>>>>>> Test:', log_str(*(dataReader.get_test_data())))





