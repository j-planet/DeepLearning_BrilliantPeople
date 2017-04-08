# # GPU Tester
#

import tensorflow as tf
from tensorflow.python.client.timeline import Timeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'  # Defaults to 0: all logs; 1: filter out INFO logs; 2: filter out WARNING; 3: filter out errors
import time

run_metadata = tf.RunMetadata()

# with tf.device("/cpu:0"):   # comment this out to use gpu instead

dim=7000

X1  = tf.get_variable("X1", shape=[dim, dim],initializer=tf.contrib.layers.xavier_initializer())
X2  = tf.get_variable("X2", shape=[dim, dim],initializer=tf.contrib.layers.xavier_initializer())
X3  = tf.get_variable("X3", shape=[dim, dim],initializer=tf.contrib.layers.xavier_initializer())
X4  = tf.get_variable("X4", shape=[dim, dim],initializer=tf.contrib.layers.xavier_initializer())

prod1 = tf.nn.tanh(tf.matmul(X1,X2))
prod2 = tf.nn.tanh(tf.matmul(X3,X4))

# prod2 = tf.assign(X1,prod1)

prod3 = tf.nn.tanh(tf.matmul(prod1,prod2))


reduced = tf.reduce_max(prod3)
#------------------------

#config.intra_op_parallelism_threads = 1
#config.inter_op_parallelism_threads = 1
#config.log_device_placement=True

# gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
# conf = tf.ConfigProto(gpu_options=gpu_opt)
# session = tf.InteractiveSession(config=conf)

session = tf.InteractiveSession()

session.run(tf.initialize_all_variables())

startTime = time.time()

for index in range(50):
    if index==1:  #reset start time due to GPU overhead
        startTime = time.time()

    print(
        session.run(reduced,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)
    )

    currentTime = time.time()
    print(index,"TotalTime", (currentTime-startTime))

trace = Timeline(step_stats=run_metadata.step_stats)
trace_file = open('~/timeline.ctf.json', 'w')
trace_file.write(trace.generate_chrome_trace_format())