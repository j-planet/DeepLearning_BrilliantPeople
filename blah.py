import glob
import json
import nltk
from pprint import pprint
import numpy as np
import collections
from sklearn.cluster import KMeans


EMBEDDINGS_FILENAME = './data/glove/glove.6B/glove.6B.50d.txt'
# EMBEDDINGS_FILENAME = './data/glove/glove.42B.300d.txt'
# EMBEDDINGS_FILENAME = './data/peopleData/earlyLifeEmbeddings.128d_alltokens.txt'


words_with_embeddings = set()

# first pass: get available words first
with open(EMBEDDINGS_FILENAME, encoding='utf8') as ifile:
    for line in ifile.readlines():
        token = line.split()[0]
        words_with_embeddings.add(token)


IGNORED_TOKENS = ['was', 'time', 'later', 'is', 'were', 'be', 'been', 'have', 'became', 'year', 'life', 'died', 'born', 'had', 'did', 'do', 'said', 'are', 'has', 'such', 'father', 'mother', 'death', 'while', 'including', 'whose', 'whom', 'known', '-', '–', "''"]
IGNORED_POS = [',', '.', ':', '``', '(', ')', '', "''",
               'IN', 'EX', 'TO', 'DT', 'WRB', 'WDT', 'WP', 'CC', 'PRP', 'PRP$', 'MD', 'POS', 'RB', 'RBR', 'RBS']

tokensByPerson = {}     # { person: [non-ignored tokens...] }
noVectorTokens = set()
allTokens = []


def knn_contexts(numContexts, vectorMat, tokens):

    assert vectorMat.shape[0] == len(tokens), vectorMat.shape

    kmeans = KMeans(n_clusters=numContexts, max_iter=1000).fit(vectorMat)
    contextByCount = []

    for i in range(numContexts):
        curTokens = np.array(tokens)[kmeans.labels_==i]
        c = curTokens[:, 1].astype(int).sum()
        tks = curTokens[:, 0]
        contextByCount.append((c, tks))

        # for tagcrowd (text visualization)
        # with open('data/peopleData/contexts/%d.txt' % i, 'w', encoding='utf8') as ofile:
        #     ofile.write(' '.join((p + ' ') * int(c) for p, c in curTokens))


    contextByCount.sort(key=lambda k: k[0])
    pprint(contextByCount)


def create_contexts(peopleNames):

    for filename in glob.glob('./data/peopleData/earlyLifes/*.txt'):    # only process those for whom we have early life texts

        person = filename.split('/')[-1].split('.')[0]

        if ' '.join(person.lower().split('_')) not in peopleNames: continue

        with open(filename, encoding='utf8') as ifile:
            text = ifile.read()

        tokensByPerson[person] = []

        taggedTokens = nltk.pos_tag(nltk.word_tokenize(text))

        for token, pos in taggedTokens:
            lowercasedToken = token.lower()

            if pos not in IGNORED_POS \
                    and lowercasedToken not in IGNORED_TOKENS \
                    and lowercasedToken not in person.lower():

                tokensByPerson[person].append(token)

                if lowercasedToken in words_with_embeddings:
                    allTokens.append(lowercasedToken)
                else:
                    noVectorTokens.add(token)


    TOKENS_TO_CONSIDER = None   # 'None' means all (how philosophical...)
    tokensChosen = collections.Counter(allTokens).most_common(TOKENS_TO_CONSIDER)


    # second pass: get vectors
    word2vecData = {}
    tokensChosen_hash = set(p[0] for p in tokensChosen)   # for fast lookup

    with open(EMBEDDINGS_FILENAME, encoding='utf8') as ifile:
        for line in ifile.readlines():

            token = line.split()[0]

            if token in tokensChosen_hash:
                word2vecData[token] = [float(d) for d in line.split()[1:]]

    del tokensChosen_hash

    X = np.array([word2vecData[t] for t in [p[0] for p in tokensChosen]])
    del word2vecData

    numContexts = 100
    knn_contexts(numContexts, X, tokensChosen)


# TODO: contexts: that contain:
# family
# rich
# school
# work
# ---> visualize!!
# TODO: remove the really common (how to define?) words from a given context to make visualization useful. TFIDF...?
#       e.g. the 'family' context will have a lot of 'son', 'children', etc, which are not useful


with open('data/peopleData/processed_names.json', encoding='utf8') as ifile:
    peoplesData = json.load(ifile)

occupations = np.unique([p[1]['occupation'][-1] for p in peoplesData.items()]) # all unique occupations
occupation = occupations[1]
create_contexts({k for (k, v) in peoplesData.items() if occupation in v['occupation']})
print(occupation)
