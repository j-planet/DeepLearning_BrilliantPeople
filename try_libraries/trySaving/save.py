import tensorflow as tf


filename = '/tmp/tfsave/save.ckpt'

sess = tf.InteractiveSession()

v1 = tf.Variable(5, name='v1')
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(5):
    sess.run(tf.assign(v1, i))
    print('Step %d: v1 = %d' % (i, i))
    saver.save(sess, filename, global_step=i)

saver.save(sess, filename)