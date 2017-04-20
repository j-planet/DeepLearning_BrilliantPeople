import tensorflow as tf

filename = '/tmp/tfsave/save.ckpt'


sess = tf.InteractiveSession()
v1 = tf.Variable(99, name='v1')
sess.run(tf.global_variables_initializer())


print('========== BEFORE restoring ==========')
for v in tf.global_variables():
    print(v.name, '----->', sess.run(v))


tf.train.Saver().restore(sess, filename)

print('========== AFTER restoring ==========')
for v in tf.global_variables():
    print(v.name, '----->', sess.run(v))
