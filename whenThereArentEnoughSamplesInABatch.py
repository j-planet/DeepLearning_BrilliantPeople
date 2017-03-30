import tensorflow as tf

import numpy as np

X = np.random.randn(2, 10, 8)
X[1, 6:] = 0
real_x_lengths = [10, 6]

outputs, last_states = tf.nn.dynamic_rnn(
    tf.contrib.rnn.LSTMCell(num_units=5, state_is_tuple=True),
    dtype=tf.float64,
    # sequence_length=real_x_lengths,
    inputs=X
)

result = tf.contrib.learn.run_n(
    {'outputs': outputs, 'last_states': last_states},
    n=1, feed_dict=None
)[0]