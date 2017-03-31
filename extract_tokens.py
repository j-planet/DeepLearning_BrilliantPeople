import string


ZERO_WIDTH_UNICODES = [u'\u200d', u'\u200c', u'\u200b']
MULTI_PERIODS = ['......', '.....', '....', '...', '…']
ALL_PUNCTUATIONS_EXCEPT_SINGLE_PERIOD = MULTI_PERIODS \
                                        + ['—', '–', '£', '€', '’', '‘', '“', 'ˈ'] \
                                        + [p for p in list(string.punctuation) if p != '.']

def extract_all_tokens(inputFilename):

    # read all possible tokens
    with open(inputFilename, encoding='utf8') as ifile:
        text = ifile.read().lower()

        for joiner in ZERO_WIDTH_UNICODES:  # remove things like <200c> (u'\u200c')
            text = text.replace(joiner, ' ')

        tokens = text.split()     # '‌' is not an empty string. It's <200c> (actually of length 1)

    tokenSet = set()

    # separate out punctuations
    for token in tokens:

        for punc in ALL_PUNCTUATIONS_EXCEPT_SINGLE_PERIOD:
            token = token.replace(punc, ' ' + punc + ' ')

        parts = token.split()

        for part in parts:  # round 2 for a single period
            if '.' in part and part not in MULTI_PERIODS:
                parts.remove(part)
                parts += part.replace('.', ' . ').split()

        tokenSet.update(parts)

    del tokens

    return tokenSet


if __name__ == '__main__':
    extractedTokens = extract_all_tokens('./data/peopleData/earlyLifeCorpus.txt')

    EMBEDDINGS_NAME_FILE = {'6B50d': './data/glove/glove.6B/glove.6B.50d.txt',
                            '6B300d': './data/glove/glove.6B/glove.6B.300d.txt',
                            '42B300d': './data/glove/glove.42B.300d.txt',
                            '840B300d': './data/glove/glove.840B.300d.txt',
                            'earlylife128d_alltokens': './data/peopleData/earlyLifeEmbeddings.128d_alltokens.txt',
                            'earlylife128d_80pc': './data/peopleData/earlyLifeEmbeddings.128d_80pc.txt',
                            'earlylife200d_alltokens': './data/peopleData/earlyLifeEmbeddings.200d_alltokens.txt',
                            'earlylife200d_80pc': './data/peopleData/earlyLifeEmbeddings.200d_80pc.txt'
                            }
    EMBEDDINGS_FILE = EMBEDDINGS_NAME_FILE['6B50d']

    embeddings = {}

    with open(EMBEDDINGS_FILE, encoding='utf8') as ifile:
        for line in ifile.readlines():
            tokens = line.split(' ')
            word = tokens[0]

            if word not in extractedTokens: continue

            vec = [float(t) for t in tokens[1:]]
            embeddings[word] = vec

    numNotFound = 0
    notFoundTokens = []

    for token in extractedTokens:
        if token not in embeddings:
            numNotFound += 1
            notFoundTokens.append(token)
            print(token, 'not found.')
    '‌'
    print('%d out of %d, or %.1f%% not found.' % (numNotFound, len(extractedTokens), 100.*numNotFound/len(extractedTokens)))