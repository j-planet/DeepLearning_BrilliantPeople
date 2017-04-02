# Code from: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py

import tensorflow as tf
from tensorflow.contrib import rnn

sess = tf.InteractiveSession()

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)

# iteration parameters
learning_rate = 1e-3
training_iters = 100000 # total number of data points (i.e. sum of step * batch_size)
batch_size = 100
print_training_stats_period = 10        # every x steps
print_validation_stats_period = 100    # every x steps

# network parameters
n_input = 28    # an image has 784 pixels, transformed into 28 sequences of length 28
n_steps = 28
n_hidden_layer_features = 128  # hidden layer number of features
n_classes = 10

# graph setup
x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

weights = tf.Variable(tf.random_normal([2*n_hidden_layer_features, n_classes]))    # *2 for forward + backward cells
biases = tf.Variable(tf.random_normal([n_classes]))

def BiRNN(x_, weights_, biases_):
    """
    :return: unscaled log probabilities
    """

    # reshape stuff
    # input shape: (batch_size x n_steps x n_input)
    # output shape: n_steps tensors, each of shape (batch_size x n_input)
    x_ = tf.split(
        tf.reshape(
            tf.transpose(x_, [1, 0, 2]),
            [-1, n_input]),
        n_steps, 0)

    # make cells
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden_layer_features, forget_bias=1.0)  # forward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden_layer_features, forget_bias=1.0)  # backward direction cell

    # get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_, dtype=tf.float32)
    except Exception:   # old TF version returns only outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_, dtype=tf.float32)

    # activation: linear, using rnn inner loop's last output
    return tf.matmul(outputs[-1], weights_) + biases_, outputs     # unscaled log probabilities


def log_str(x_, y_):
    """
    :return a string of loss and accuracy
    """

    return 'loss = %.3f, accuracy = %.3f' % \
           tuple(sess.run([cost, accuracy], feed_dict={x: x_, y: y_}))


pred, outputs = BiRNN(x, weights, biases)    # unscaled log probabilities

# define loss and the optimizer to minimize it
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialize the variables
sess.run(tf.global_variables_initializer())

# train!!!!
step = 1
x_shape = (-1, n_steps, n_input)
valid_x = mnist.validation.images.reshape(x_shape)
valid_y = mnist.validation.labels

while step * batch_size < training_iters:

    # fetch batch data
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape(x_shape)
    feed_dict = {x: batch_x, y: batch_y}

    # train
    sess.run(optimizer, feed_dict)
    sess.run(outputs, feed_dict)

    # print evaluations
    if step % print_training_stats_period == 0:
        print('Step %d, %d data points Training:' % (step, step*batch_size))
        print('Training:', log_str(batch_x, batch_y))

        if step % print_validation_stats_period == 0:   # assumes training freq is a multiple of valid freq
            print('Validation:', log_str(valid_x, valid_y), '\n')

    step += 1

# calculate test accuracy
print('\n----- TEST SET RESULTS:', log_str(mnist.test.images.reshape(x_shape), mnist.test.labels))




