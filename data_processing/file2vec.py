import string
import numpy as np


ZERO_WIDTH_UNICODES = [u'\u200d', u'\u200c', u'\u200b']
MULTI_PERIODS = ['......', '.....', '....', '...', '…']
ALL_PUNCTUATIONS_EXCEPT_SINGLE_PERIOD = MULTI_PERIODS \
                                        + ['—', '–', '£', '€', '’', '‘', '“', 'ˈ'] \
                                        + [p for p in list(string.punctuation) if p != '.']


def extract_tokenset_from_file(inputFilename):
    """ 
    :return: a set 
    """

    with open(inputFilename, encoding='utf8') as ifile:
        text = ' '.join(line.strip() for line in ifile.readlines())      # removes the new line character

    return set(insert_spaces_into_corpus(text).split())


def insert_spaces_into_corpus(text_):

    text_ = text_.lower()

    for joiner in ZERO_WIDTH_UNICODES:  # remove things like <200c> (u'\u200c')
        text_ = text_.replace(joiner, ' ')

    res = ''

    for token in text_.split():

        for punc in ALL_PUNCTUATIONS_EXCEPT_SINGLE_PERIOD:
            token = token.replace(punc, ' ' + punc + ' ')

        parts = token.split()

        for part in parts:  # round 2 for a single period

            if '.' in part and part not in MULTI_PERIODS:
                res += ' '.join(part.replace('.', ' . ').split()) + ' '

            else:
                res += part + ' '

    return res


def extract_embedding(embeddingsFilename_, relevantTokens_, includeUnk_ = True,
                      verbose = True):

    res = {}

    if includeUnk_:
        if type(relevantTokens_)==set:
            relevantTokens_.add('unk')
        elif type(relevantTokens_)==list:
            relevantTokens_.append('unk')
        else:
            raise Exception('Unknown input tokens type:', type(relevantTokens_))

    with open(embeddingsFilename_, encoding='utf8') as ifile:
        for line in ifile.readlines():
            tokens = line.split(' ')
            word = tokens[0]

            if word not in relevantTokens_: continue

            vec = [float(t) for t in tokens[1:]]
            res[word] = vec

    numNotFound = 0
    notFoundTokens = set()

    for token in relevantTokens_:
        if token not in res:
            numNotFound += 1
            notFoundTokens.add(token)

            if verbose: print(token, 'not found.')

    print('%d out of %d, or %.1f%% not found.' % (numNotFound, len(relevantTokens_), 100. * numNotFound / len(relevantTokens_)))

    return res, notFoundTokens


def file2vec(filename, embeddings_):
    with open(filename, encoding='utf8') as ifile:
        text = ' '.join(line.strip() for line in ifile.readlines())

    return np.array([embeddings_.get(token, embeddings_['unk']) for token in insert_spaces_into_corpus(text).split()])


def create_custom_embeddings_file(inputEmbeddingFilename, tokensFilename, outputFilename):

    # How to encode tokens that are missing from the embeddings file? Use 'unk' (which exists in the Glove embeddings file) for now
    embeddings, _ = extract_embedding(
        embeddingsFilename_=inputEmbeddingFilename,
        relevantTokens_=extract_tokenset_from_file(tokensFilename),
        includeUnk_=True
    )

    with open(outputFilename, 'w', encoding='utf8') as outputFile:
        for k, v in embeddings.items():
            outputFile.write(k + ' ' + ' '.join([str(n) for n in v]) + '\n')


if __name__ == '__main__':

    EMBEDDINGS_NAME_FILE = \
        {'6B50d': '../data/glove/glove.6B/glove.6B.50d.txt',
         '6B300d': '../data/glove/glove.6B/glove.6B.300d.txt',
         '42B300d': '../data/glove/glove.42B.300d.txt',
         '840B300d': '../data/glove/glove.840B.300d.txt',
         'earlylife128d_alltokens': '../data/peopleData/earlyLifeEmbeddings.128d_alltokens.txt',
         'earlylife128d_80pc': '../data/peopleData/earlyLifeEmbeddings.128d_80pc.txt',
         'earlylife200d_alltokens': '../data/peopleData/earlyLifeEmbeddings.200d_alltokens.txt',
         'earlylife200d_80pc': '../data/peopleData/earlyLifeEmbeddings.200d_80pc.txt'
         }

    # How to encode tokens that are missing from the embeddings file? Use 'unk' (which exists in the Glove embeddings file) for now
    # embeddings, _ = extract_embedding(
    #     embeddingsFilename_=EMBEDDINGS_NAME_FILE['6B50d'],
    #     relevantTokens_=extract_tokenset_from_file('../data/peopleData/earlyLifeCorpus.txt'),
    #     includeUnk_=True
    # )

    # print(file2vec('data/peopleData/earlyLifes/abbe pierre.txt', embeddings))
    create_custom_embeddings_file(EMBEDDINGS_NAME_FILE['42B300d'],
                                  '../data/peopleData/earlyLifeCorpus.txt',
                                  '../data/peopleData/embeddings/smallGlove.42B300d.txt')





