# reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

import collections
import math
import os, glob
import random
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import numpy as np
import urllib
import tensorflow as tf
from tensorflow import placeholder, Variable, reduce_mean, reduce_sum, matmul, summary

##### step 0: get tokens data (e.g. download data from http://mattmahoney.net/dc/text8.zip or make your own corpus)
CORPUS_FILENAME = 'data/peopleData/earlyLifeCorpus.txt'                # input file
EMBEDDING_SIZE = 128
EMBEDDING_FILENAME = './data/peopleData/earlyLifeEmbeddings.%dd_80pc.txt' % EMBEDDING_SIZE   # output file

##### step 1: read data
with open(CORPUS_FILENAME, encoding='utf8') as ifile:
    words = ifile.read().split()
print('Data size: %d non-unique, %d unique words' % (len(words), len(np.unique(words))))

##### step 2: build the dictionary -> replace rare words with UNK token
# VOCAB_SIZE = len(np.unique(words)) + 1  # +1 for the UNK token
VOCAB_SIZE = int(len(np.unique(words)) * 0.8)   # don't use the rare words (most likely people's names in this case)


def build_dataset(words, vocabulary_size):
    '''
    :param words:
    :param vocabulary_size: if None takes all data
    :return: data, count, dictionary, reverse_dictionary
    '''

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()
    for word, _ in count: dictionary[word] = len(dictionary)

    data = list()   # convert word to ID
    unk_count = 0
    for word in words:
        if word in dictionary:  # not a rare word
            index = dictionary[word]    # convert word to int id
        else:
            index = 0   # index = 0 <==> 'UNK'
            unk_count += 1

        data.append(index)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words, VOCAB_SIZE)
del words   # save memory
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

##### step 3: generate a training batch for the skip-gram model

def generate_batch(batch_size, num_skips, skip_window):
    '''
    :param batch_size:
    :param num_skips: How many times to re-use an input to generate a label
    :param skip_window: How many words to consider left and right
    :return:
    '''
    global data_index
    assert batch_size % num_skips == 0, 'batch_size & num_skips == 0'
    assert num_skips <= 2 * skip_window, 'num_skips <= 2 * skip_window'

    TOTAL_DATA_LEN = len(data)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window]
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % TOTAL_DATA_LEN

    for i in range(batch_size // num_skips):
        target = skip_window    # target label at the center of the buffer
        targets_to_avoid = [skip_window]

        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % TOTAL_DATA_LEN

    # backtrack a little to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % TOTAL_DATA_LEN

    return batch, labels

batch, labels = generate_batch(batch_size = 50, num_skips = 5, skip_window = 5)

print('---- sample target -> neighbor')
for i in range(10):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


##### step 4: Build and train a skip-gram model
batch_size = 128
embedding_size = EMBEDDING_SIZE    # dimension of the embedding vector
skip_window = 1         # How many words to consider left and right
num_skips = 2           # How many times to re-use an input to generate a label

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # random set of words for validation (i.e. evaluate similarity on)
valid_window = 100  # pick from the 100 most common words
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # number of NEGATIVE examples to sample

# graph = tf.Graph()
sess = tf.InteractiveSession()

# define input data
train_inputs = placeholder(tf.int32, shape=[batch_size])
train_labels = placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Ops and variables pinned to the CPU because of missing GPU implementation
with tf.device('/cpu:0'):
    # initialize embeddings to uniform randoms, used for looking up embeddings for inputs
    embeddings = Variable(tf.random_uniform([VOCAB_SIZE, embedding_size], -1., 1.))    # ~Unif(-1, 1)
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # construct the variables for the NCE loss
    # nce_weights ~ N(0, 1/sqrt(embedding size)) of size vocab_size x embedding_size
    nce_weights = Variable(tf.truncated_normal([VOCAB_SIZE, embedding_size], stddev=1. / math.sqrt(embedding_size)))
    # nce_biases ~ vector of zeros of size vocab_size
    nce_biases = Variable(tf.zeros([VOCAB_SIZE]))

# define loss function
# tf.nn.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss
loss = reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases,
                                  inputs = embed, labels = train_labels,
                                  num_sampled=num_sampled,
                                  num_classes=VOCAB_SIZE))
summary.scalar('loss', loss)


# define the optimizer
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# compute the cosine similarity between minibatch examples and all embeddings
normalized_embeddings = embeddings / tf.sqrt(reduce_sum(tf.square(embeddings), 1, keep_dims=True))
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


##### step 5: train!
# set up tensorboard
merged_summaries = summary.merge_all()
for f in glob.glob('./logs/word2vec/train/*'): os.remove(f)
for f in glob.glob('./logs/word2vec/validation/*'): os.remove(f)
train_writer = summary.FileWriter('./logs/word2vec/train', sess.graph)
valid_writer = summary.FileWriter('./logs/word2vec/validation')
sess.run(tf.global_variables_initializer())

num_steps = 100001
report_period = 100
validation_period = 5000   # Note that this is expensive (~20% slowdown if computed every 500 steps)
average_loss = 0    # average loss per "reporting" period

for step in range(num_steps):

    batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op
    _, loss_val, summaries = sess.run([optimizer, loss, merged_summaries], feed_dict=feed_dict)
    average_loss += loss_val

    if step % report_period == 0:
        if step > 0: average_loss /= report_period
        train_writer.add_summary(summaries, step)
        print('\nAverage loss at step', step, ':', average_loss)
        average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % validation_period == 0:
        sim = similarity.eval()
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8   # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log_str = 'Top %d Nearest to %s: %s' % (top_k,
                                                    valid_word,
                                                    ' '.join(reverse_dictionary[nearest[k]] for k in range(top_k)))
            print(log_str)

final_embeddings = normalized_embeddings.eval()
tokens = [p[1] for p in sorted(reverse_dictionary.items(), key=lambda p: p[0]) if p[1]!='UNK']

train_writer.flush()
valid_writer.flush()

##### output to file
with open(EMBEDDING_FILENAME, 'w', encoding='utf8') as ofile:
    for i, token in enumerate(tokens):
        ofile.write('%s %s\n' % (token, ' '.join(str(d) for d in final_embeddings[i, :])))






























