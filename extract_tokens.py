import string


ZERO_WIDTH_UNICODES = [u'\u200d', u'\u200c', u'\u200b']
MULTI_PERIODS = ['......', '.....', '....', '...', '…']
ALL_PUNCTUATIONS_EXCEPT_SINGLE_PERIOD = MULTI_PERIODS \
                                        + ['—', '–', '£', '€', '’', '‘', '“', 'ˈ'] \
                                        + [p for p in list(string.punctuation) if p != '.']


def extract_tokenset_from_file(inputFilename):
    """ 
    :return: a set 
    """

    # read all possible tokens
    with open(inputFilename, encoding='utf8') as ifile:
        text = ifile.read()

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


def extract_relevant_embedding(embeddingsFilename, relevantTokens):

    embeddings = {}

    with open(embeddingsFilename, encoding='utf8') as ifile:
        for line in ifile.readlines():
            tokens = line.split(' ')
            word = tokens[0]

            if word not in relevantTokens: continue

            vec = [float(t) for t in tokens[1:]]
            embeddings[word] = vec

    numNotFound = 0
    notFoundTokens = []

    for token in relevantTokens:
        if token not in embeddings:
            numNotFound += 1
            notFoundTokens.append(token)
            print(token, 'not found.')

    print('%d out of %d, or %.1f%% not found.' % (numNotFound, len(relevantTokens), 100.*numNotFound/len(relevantTokens)))

    return embeddings


if __name__ == '__main__':

    EMBEDDINGS_NAME_FILE = {'6B50d': './data/glove/glove.6B/glove.6B.50d.txt',
                            '6B300d': './data/glove/glove.6B/glove.6B.300d.txt',
                            '42B300d': './data/glove/glove.42B.300d.txt',
                            '840B300d': './data/glove/glove.840B.300d.txt',
                            'earlylife128d_alltokens': './data/peopleData/earlyLifeEmbeddings.128d_alltokens.txt',
                            'earlylife128d_80pc': './data/peopleData/earlyLifeEmbeddings.128d_80pc.txt',
                            'earlylife200d_alltokens': './data/peopleData/earlyLifeEmbeddings.200d_alltokens.txt',
                            'earlylife200d_80pc': './data/peopleData/earlyLifeEmbeddings.200d_80pc.txt'
                            }

    relevantTokens = extract_tokenset_from_file('./data/peopleData/earlyLifeCorpus.txt').union({'unk'})

    # How to encode tokens that are missing from the embeddings file? Use 'unk' (which exists in the Glove embeddings file) for now
    embeddings = extract_relevant_embedding(
        embeddingsFilename=EMBEDDINGS_NAME_FILE['6B50d'],
        relevantTokens=extract_tokenset_from_file('./data/peopleData/earlyLifeCorpus.txt').union({'unk'}))



    file = open('data/peopleData/earlyLifes/Abbe_Pierre.txt', encoding='utf8')
    lines = file.readlines()
    line = lines[0]






