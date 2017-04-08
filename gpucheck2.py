import tensorflow as tf

no_opt = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0,
                             do_common_subexpression_elimination=False,
                             do_function_inlining=False,
                             do_constant_folding=False)
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=no_opt),
                        log_device_placement=True, allow_soft_placement=False,
                        device_count={"CPU": 3},
                        inter_op_parallelism_threads=3,
                        intra_op_parallelism_threads=1)
sess = tf.Session(config=config)

with tf.device("cpu:0"):
    a = tf.ones((13, 1))

sess = tf.Session(config=config)
run_metadata = tf.RunMetadata()
sess.run(a, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)
with open("/tmp/run2.txt", "w") as out:
    out.write(str(run_metadata))