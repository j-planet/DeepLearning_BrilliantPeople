import tensorflow as tf
from multiprocessing import cpu_count
import json

from train import RunConfig
from data_readers.embedding_data_reader import EmbeddingDataReader
from models.mark6 import Mark6, RNNConfig


runConfig = RunConfig('medium')

# make data reader
dataReaderMaker = EmbeddingDataReader.maker_from_premade_source('full')
dataReader = dataReaderMaker(bucketingOrRandom='bucketing', batchSize_=runConfig.batchSize, minimumWords=40)

# make model
p = dict([('initialLearningRate', 1e-3),
          ('l2RegLambda', 1e-6),
          ('l2Scheme', 'overall'),

          ('rnnConfigs', [RNNConfig([1024, 512], [0.6, 0.7])]),

          ('pooledKeepProb', 0.9),
          ('pooledActivation', None)
          ])

modelMaker = lambda input_, logFac: Mark6(input_=input_, **p, loggerFactory_=logFac)
model = modelMaker(dataReader.input, None)

# make session
numCores = cpu_count() - 1
config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=numCores, inter_op_parallelism_threads=numCores)

with tf.device('/cpu:0'):
    sess = tf.InteractiveSession(config=config)

# restore from saved path
savePath = '/Users/jj/Code/brilliant_people/logs/main/Mark6/loadmark6/saved/save.ckpt'
tf.train.Saver().restore(sess, savePath)

# feed in some data and evaluate

res = [[ (model.evaluate(sess, fd, full=True), names) for
         fd, names in bg ]
       for bg in [dataReader.get_validation_data_in_batches(), dataReader.get_test_data_in_batches()]]

with open('/Users/jj/Code/brilliant_people/logs/main/Mark6/metrics.json', 'w') as ofile:
    json.dump(res, ofile)