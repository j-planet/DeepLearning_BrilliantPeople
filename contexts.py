import glob, os, pathlib
import json
import nltk
from pprint import pprint
import numpy as np
import collections
from sklearn.cluster import KMeans


IGNORED_TOKENS = ['was', 'time', 'later', 'is', 'were', 'be', 'been', 'have', 'became', 'year', 'life', 'died', 'born', 'had', 'did', 'do', 'said', 'are', 'has', 'such', 'father', 'mother', 'death', 'while', 'including', 'whose', 'whom', 'known', '-', '–', "''"]
IGNORED_POS = [',', '.', ':', '``', '(', ')', '', "''",
               'IN', 'EX', 'TO', 'DT', 'WRB', 'WDT', 'WP', 'CC', 'PRP', 'PRP$', 'MD', 'POS', 'RB', 'RBR', 'RBS']


def knn_contexts(numContexts, vectorMat, tokens, outputDir=None):

    assert vectorMat.shape[0] == len(tokens), vectorMat.shape

    kmeans = KMeans(n_clusters=numContexts, max_iter=1000).fit(vectorMat)
    contextByCount = []

    for i in range(numContexts):
        curTokens = np.array(tokens)[kmeans.labels_==i]
        c = curTokens[:, 1].astype(int).sum()
        tks = curTokens[:, 0]
        contextByCount.append((c, tks))

        # for tagcrowd (text visualization)
        if outputDir:
            with open('%s/%d.txt' % (outputDir, i), 'w', encoding='utf8') as ofile:
                ofile.write(' '.join((p + ' ') * int(c) for p, c in curTokens))


    contextByCount.sort(key=lambda k: k[0])
    pprint(contextByCount)


def read_word2vec_data(peopleNames, embeddingsFilename, extraTokens = [], numMostCommonTokens = None):
    '''
    :param peopleNames: ['abraham lincoln', ...]
    :param extraTokens: tokens to get vectors for, even if they are not in the corpuses (corpi...? corpa...?)
    :param numMostCommonTokens: return only the x most common tokens. if None returns all.
    :return: word2vec vectors dictionary,
            chosenTokensNCounts (list),
            allVectorsMat (2D np.array), extraTokensMat (2D np.array),
            noVectorTokens (set)
            Note: chosenTokensNCounts correspond to the columns of allVectorsMat
    '''

    words_with_embeddings = set()

    # first pass: get available words first
    with open(embeddingsFilename, encoding='utf8') as ifile:
        for line in ifile.readlines():
            token = line.split()[0]
            words_with_embeddings.add(token)

    tokensByPerson = {}  # { person: [non-ignored tokens...] }
    noVectorTokens = set()
    allTokens = []

    numTotalTokens = 0
    numIgnored = 0
    numNoVector = 0

    for filename in glob.glob('./data/peopleData/earlyLifesTexts/*.txt'):    # only process those for whom we have early life texts

        person = filename.split('/')[-1].split('.')[0]

        if ' '.join(person.lower().split('_')) not in peopleNames: continue

        with open(filename, encoding='utf8') as ifile:
            text = ifile.read()

        tokensByPerson[person] = []

        taggedTokens = nltk.pos_tag(nltk.word_tokenize(text))
        numTotalTokens += len(taggedTokens)

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
                    numNoVector += 1
            else:
                numIgnored += 1

    print('>>> Out of a total of %d tokens (non-uniqued), %d are ignored, %d have no vectors.'
          % (numTotalTokens, numIgnored, numNoVector))
    print('Tokens with no existing vectors:', noVectorTokens, '\n')

    if type(numMostCommonTokens) == float:
        numMostCommonTokens = int(len(np.unique(allTokens)) * numMostCommonTokens)

    chosenTokensNCounts = collections.Counter(allTokens).most_common(numMostCommonTokens)


    # second pass: get vectors

    for extraToken in extraTokens:
        if extraToken not in [p[0] for p in chosenTokensNCounts]:
            chosenTokensNCounts.append((extraToken, 0))

    tokensChosen_hash = set(p[0] for p in chosenTokensNCounts)   # for fast lookup

    word2vecData = {}
    with open(embeddingsFilename, encoding='utf8') as ifile:
        for line in ifile.readlines():

            token = line.split()[0]

            if token in tokensChosen_hash:
                try:
                    word2vecData[token] = [float(d) for d in line.split()[1:]]
                except:
                    pass

    del tokensChosen_hash

    allVectorsMat = np.array([word2vecData[t] for t in [p[0] for p in chosenTokensNCounts]])
    extraTokensMat = np.array([word2vecData[p] for p in extraTokens])

    return word2vecData, chosenTokensNCounts, allVectorsMat, extraTokensMat, noVectorTokens


def read_people_names(inputFilename ='data/peopleData/processed_names.json', occupation=None):
    '''
    :param occupation: if None, returns all occupations
    :return: set of people names
    '''

    with open(inputFilename, encoding='utf8') as ifile:
        peoplesData = json.load(ifile)

    return {k for (k, v) in peoplesData.items() if occupation in v['occupation']} if occupation \
        else {k for (k, v) in peoplesData.items()}


def read_occupations(inputFilename ='data/peopleData/processed_names.json'):
    with open(inputFilename, encoding='utf8') as ifile:
        peoplesData = json.load(ifile)

    return np.unique([p[1]['occupation'][-1] for p in peoplesData.items()])  # all unique occupations


def keyword_contexts(numTopSimilarTokens, clusterKeywords,
                     tokensNCounts, allVectorsMat, keywordVectorsMat,
                     outputDir=None):
    '''
    :param numTopSimilarTokens: if None, takes all tokens
    :param outputDir: if None, does not output any files
    :param occupation: if None, all occupations LUMPED together
    :return:
    '''

    assert numTopSimilarTokens is not None, \
        'Taking all tokens means the outcome will be the same for all keywords, defeating the purpose of this function.'

    similarityMatrix = np.matmul(keywordVectorsMat, allVectorsMat.T)

    for i, clusterKeyword in enumerate(clusterKeywords):

        topIndices = (-similarityMatrix[i, :]).argsort()[:(numTopSimilarTokens+1)] if numTopSimilarTokens \
            else (-similarityMatrix[i, :]).argsort()

        topTokens = np.array(tokensNCounts)[topIndices]

        assert topTokens[0][0] == clusterKeyword, \
            'A word must have the highest similarity with itself. %s vs %s' %(topTokens[0][0], clusterKeyword)

        print(clusterKeyword, ':\n', topTokens[1:], '\n')

        # write to file
        pathlib.Path(outputDir).mkdir(parents=True, exist_ok=True)

        with open(outputDir + '/' + clusterKeyword + '.txt', 'w', encoding='utf8') as ofile:
            tagwordText = ' '.join((t + ' ') * int(c) for t, c in topTokens[1:])
            ofile.write(tagwordText)



if __name__ == '__main__':
    EMBEDDINGS_NAME_FILE = {'6B300d': './data/glove/glove.6B/glove.6B.300d.txt',
                            '42B300d': './data/glove/glove.42B.300d.txt',
                            '840B300d': './data/glove/glove.840B.300d.txt',
                            'earlylife128d_alltokens': './data/peopleData/earlyLifeEmbeddings.128d_alltokens.txt',
                            'earlylife128d_80pc': './data/peopleData/earlyLifeEmbeddings.128d_80pc.txt',
                            'earlylife200d_alltokens': './data/peopleData/earlyLifeEmbeddings.200d_alltokens.txt',
                            'earlylife200d_80pc': './data/peopleData/earlyLifeEmbeddings.200d_80pc.txt'
                            }
    EMBEDDINGS_NAME = '840B300d'


    # =========== keyword-based context clusters ===========
    clusterKeywords = ['school', 'work', 'family', 'money', 'love', 'rich', 'poor']

    for occupation in [None] + list(read_occupations()):
        print('\n', '>'*10, occupation)

        _, tokensNCounts, allVectorsMat, keywordVectorsMat, _ = \
            read_word2vec_data(read_people_names(occupation=occupation),
                               embeddingsFilename=EMBEDDINGS_NAME_FILE[EMBEDDINGS_NAME],
                               extraTokens=clusterKeywords,
                               numMostCommonTokens=None)

        keyword_contexts(50, clusterKeywords,
                         tokensNCounts, allVectorsMat, keywordVectorsMat,
                         outputDir='data/peopleData/contexts/keyword_contexts/%s/%s' % (EMBEDDINGS_NAME, occupation or 'ALL'))


    # =========== KNN context clusters ===========
    # for occupation in [None] + list(read_occupations()):
    #     print('\n', '>'*10, occupation)

    # _, tokensNCounts, allVectorsMat, _, _ = read_word2vec_data(read_people_names(occupation=None))
    # numContexts = 100
    # knn_contexts(numContexts, allVectorsMat, tokensNCounts, 'data/peopleData/contexts/knn_contexts')

