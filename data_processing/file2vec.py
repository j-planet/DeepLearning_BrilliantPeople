import string
import os, glob
import json
import numpy as np
import difflib


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
    """
    :param relevantTokens_: if None, then all lines in the embeddings file are returned 
    :return: (dictionary, list of not found tokens if releventTokens is not None; otherwise, None) 
    """

    res = {}
    useRelevantTokens = relevantTokens_ is not None

    if includeUnk_ and useRelevantTokens:
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

            if useRelevantTokens and word not in relevantTokens_: continue

            vec = [float(t) for t in tokens[1:]]
            res[word] = vec

    numNotFound = 0

    if useRelevantTokens:
        notFoundTokens = set()

        for token in relevantTokens_:
            if token not in res:
                numNotFound += 1
                notFoundTokens.add(token)

                if verbose: print(token, 'not found.')

        print('%d out of %d, or %.1f%% not found.' % (numNotFound, len(relevantTokens_), 100. * numNotFound / len(relevantTokens_)))
    else:
        notFoundTokens = None

    return res, notFoundTokens


def file2vec(filename, embeddings_, outputFilename = None):
    with open(filename, encoding='utf8') as ifile:
        text = ' '.join(line.strip() for line in ifile.readlines())

    mat = np.array([embeddings_.get(token, embeddings_['unk']) for token in insert_spaces_into_corpus(text).split()])

    if outputFilename:
        with open(outputFilename, 'w', encoding='utf8') as ofile:
            json.dump({'occupation': occupation, 'mat': [list(l) for l in list(mat)]}, ofile)

    return mat


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

    PPL_DATA_DIR = '../data/peopleData'

    EMBEDDINGS_NAME_FILE = \
        {'6B50d': '../data/glove/glove.6B/glove.6B.50d.txt',
         '6B300d': '../data/glove/glove.6B/glove.6B.300d.txt',
         '42B300d': '../data/glove/glove.42B.300d.txt',
         '840B300d': '../data/glove/glove.840B.300d.txt',
         'earlylife128d_alltokens': '../data/peopleData/embeddings/earlyLifeEmbeddings.128d_alltokens.txt',
         'earlylife128d_80pc': '../data/peopleData/embeddings/earlyLifeEmbeddings.128d_80pc.txt',
         'earlylife200d_alltokens': '../data/peopleData/embeddings/earlyLifeEmbeddings.200d_alltokens.txt',
         'earlylife200d_80pc': '../data/peopleData/embeddings/earlyLifeEmbeddings.200d_80pc.txt'
         }

    embeddingName = '42B300d'

    embeddings, _ = extract_embedding(
        embeddingsFilename_=EMBEDDINGS_NAME_FILE[embeddingName],
        relevantTokens_=None,
        includeUnk_=True
    )
    print('DONE reading embeddings.')

    # read { name: occupation }
    peopleData = {}
    for filename in glob.glob(os.path.join(PPL_DATA_DIR, 'processed_names/*.json')):
        with open(filename, encoding='utf8') as ifile:
            peopleData.update(json.load(ifile))


    processed = 0
    outputDir = os.path.join(PPL_DATA_DIR, 'earlyLifesWordMats_' + embeddingName)
    if not os.path.exists(outputDir): os.mkdir(outputDir)

    for filename in glob.glob(os.path.join(PPL_DATA_DIR, 'earlyLifesTexts/*.txt')):

        name = filename.split('/')[-1].split('.txt')[0]
        outputFname = os.path.join(outputDir, name + '.json')

        if os.path.exists(outputFname):
            print(outputFname, 'already exists. Skipping...')
            continue

        if name in peopleData:
            occupation = peopleData[name]['occupation']
        else:
            fuzzyMatches = difflib.get_close_matches(name, peopleData.keys(), 1)   # hack for the issue of matching utf8 names

            if fuzzyMatches:
                fuzzyMatchedName = fuzzyMatches[0]
                occupation = peopleData[fuzzyMatchedName]['occupation']
                print('FUZZY MATCH: %s -> %s' % (name, fuzzyMatchedName))
            else:
                print('ERROR: %s does not exist in names json files. Not even a fuzzy match.' % name)
                continue

        mat = file2vec(filename, embeddings)

        if len(mat)==0:
            print('ERROR: no content read for %s. Skipping...' % name)
            continue

        with open(outputFname, 'w', encoding='utf8') as ofile:
            json.dump({'occupation': occupation, 'mat': [list(l) for l in list(mat)]}, ofile)

        processed += 1

    print('Processed %d out of %d in names json files.' % (processed, len(peopleData)))