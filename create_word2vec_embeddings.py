# reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

import collections
import math
import os, glob
import random

import numpy as np
import tensorflow as tf
from tensorflow import placeholder, Variable, reduce_mean, reduce_sum, matmul, summary

from blah import read_occupations

data_index = 0

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

def generate_batch(data, batchSize, numSkips, skipWindow):
    '''
    :param data: list of IDs
    :param batchSize:
    :param numSkips: How many times to re-use an input to generate a label
    :param skipWindow: How many words to consider left and right
    :return:
    '''
    global data_index
    assert batchSize % numSkips == 0, 'batch_size & num_skips == 0'
    assert numSkips <= 2 * skipWindow, 'num_skips <= 2 * skipWindow'

    TOTAL_DATA_LEN = len(data)
    batch = np.ndarray(shape=(batchSize), dtype=np.int32)
    labels = np.ndarray(shape=(batchSize, 1), dtype=np.int32)
    span = 2 * skipWindow + 1  # [ skipWindow target skipWindow ]
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % TOTAL_DATA_LEN

    for i in range(batchSize // numSkips):
        target = skipWindow    # target label at the center of the buffer
        targets_to_avoid = [skipWindow]

        for j in range(numSkips):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i * numSkips + j] = buffer[skipWindow]
            labels[i * numSkips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % TOTAL_DATA_LEN

    # backtrack a little to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % TOTAL_DATA_LEN

    return batch, labels

def create_embeddings(corpusFilename, outputFilename,
                      vocabSize, batchSize, embeddingDimension, numNegativeExamples, numSteps,
                      validationExamples,
                      skipWindow = 1, numSkips = 2):
    '''
    :param corpusFilename: 
    :param outputFilename: 
    :param batchSize: 
    :type batchSize: int
    :param embeddingDimension: 
    :param numNegativeExamples: 
    :param validationExamples: if int, chosen at random from the 100 most common words; if list (of string tokens), used for validation directly
    :param skipWindow: 
    :param numSkips: 
    :return: 
    '''

    print('SETUP: batchSize: %d, embeddingDimension: %d, numNegativeExamples: %d, numSteps: %d, corpusFilename: %s' %
          (batchSize, embeddingDimension, numNegativeExamples, numSteps, corpusFilename))

    global data_index
    data_index = 0
    tf.reset_default_graph()

    ##### step 0: get tokens data (e.g. download data from http://mattmahoney.net/dc/text8.zip or make your own corpus)

    ##### step 1: read data
    with open(corpusFilename, encoding='utf8') as ifile:
        words = ifile.read().split()
    print('Data size: %d non-unique, %d unique words' % (len(words), len(np.unique(words))))

    ##### step 2: build the dictionary -> replace rare words with UNK token
    # VOCAB_SIZE = len(np.unique(words)) + 1  # +1 for the UNK token
    # vocabSize = int(len(np.unique(words)) * 0.8)  # don't use the rare words (most likely people's names in this case)

    if type(vocabSize)==float:  # a portion
        vocabSize = int(len(np.unique(words)) * vocabSize)
    elif type(vocabSize)==None: # max
        vocabSize = len(np.unique(words)) + 1

    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabSize)
    del words   # save memory
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


    ##### step 3: generate a training batch for the skip-gram model
    batch, labels = generate_batch(data, batchSize=50, numSkips=5, skipWindow= 5)

    print('---- sample target -> neighbor')
    for i in range(10):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


    ##### step 4: Build and train a skip-gram model

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.

    if type(validationExamples)==int:
        validationExamples = np.random.choice(100, validationExamples, replace=False)
    else:
        for w in validationExamples:
            if w not in dictionary:
                print('>>>>> Validation word %s is NOT in the corpus!' % w)

        validationExamples = [dictionary[w] for w in validationExamples if w in dictionary]

    # graph = tf.Graph()
    sess = tf.InteractiveSession()

    # define input data
    train_inputs = placeholder(tf.int32, shape=[batchSize])
    train_labels = placeholder(tf.int32, shape=[batchSize, 1])
    valid_dataset = tf.constant(validationExamples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # initialize embeddings to uniform randoms, used for looking up embeddings for inputs
        embeddings = Variable(tf.random_uniform([vocabSize, embeddingDimension], -1., 1.))    # ~Unif(-1, 1)
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # construct the variables for the NCE loss
        # nce_weights ~ N(0, 1/sqrt(embedding size)) of size vocab_size x embedding_size
        nce_weights = Variable(tf.truncated_normal([vocabSize, embeddingDimension], stddev=1. / math.sqrt(embeddingDimension)))
        # nce_biases ~ vector of zeros of size vocab_size
        nce_biases = Variable(tf.zeros([vocabSize]))

    # define loss function
    # tf.nn.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss
    loss = reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases,
                                      inputs = embed, labels = train_labels,
                                      num_sampled=numNegativeExamples,
                                      num_classes=vocabSize))
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

    reportPeriod = max(int(0.01 * numSteps), 100)
    validation_period = min(5000, int(0.05 * numSteps))   # Note that this is expensive (~20% slowdown if computed every 500 steps)
    average_loss = 0    # average loss per "reporting" period

    for step in range(numSteps):

        batch_inputs, batch_labels = generate_batch(data, batchSize, numSkips, skipWindow)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op
        _, loss_val, summaries = sess.run([optimizer, loss, merged_summaries], feed_dict=feed_dict)
        average_loss += loss_val

        if step % reportPeriod == 0:
            if step > 0: average_loss /= reportPeriod
            train_writer.add_summary(summaries, step)
            print('\nAverage loss at step', step, ':', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % validation_period == 0:
            sim = similarity.eval()
            for i, validationExample in enumerate(validationExamples):
                valid_word = reverse_dictionary[validationExample]
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
    if outputFilename:
        with open(outputFilename, 'w', encoding='utf8') as ofile:
            for i, token in enumerate(tokens):
                ofile.write('%s %s\n' % (token, ' '.join(str(d) for d in final_embeddings[i, :])))


EMBEDDING_SIZE = 200

for occupation in read_occupations():
    # occupation = 'scientist'
    corpusFilename = 'data/peopleData/earlyLifeCorpus_%s.txt' % occupation
    # corpusFilename = 'data/text8.txt'

    if os.path.exists('./data/peopleData/earlyLifeEmbeddings.%dd_80pc_%s.txt' % (EMBEDDING_SIZE, occupation)):
        print('already done %s, moving on.' % occupation)

    create_embeddings(corpusFilename=corpusFilename,
                      outputFilename='./data/peopleData/earlyLifeEmbeddings.%dd_80pc_%s.txt' % (EMBEDDING_SIZE, occupation),
                      vocabSize=0.8, batchSize=700, embeddingDimension=EMBEDDING_SIZE, numNegativeExamples=32,
                      numSteps = 30001,
                      validationExamples=['school', 'work', 'family', 'money', 'love', 'rich', 'poor', 'life', 'friend']
                      )