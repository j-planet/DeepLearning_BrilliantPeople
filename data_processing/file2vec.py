import string
import os, glob
import json
import numpy as np
import difflib


PPL_DATA_DIR = '../data/peopleData'
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


def file2vec(filename, embeddings_, occupation_, outputFilename = None):
    with open(filename, encoding='utf8') as ifile:
        text = ' '.join(line.strip() for line in ifile.readlines())

    mat = np.array([embeddings_.get(token, embeddings_['unk']) for token in insert_spaces_into_corpus(text).split()])

    if outputFilename:
        with open(outputFilename, 'w', encoding='utf8') as ofile:
            json.dump({'occupation': occupation_, 'mat': [list(l) for l in list(mat)]}, ofile)

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


def filename2name(filename_):
    return os.path.splitext(os.path.basename(filename_))[0]

class OccupationReader(object):

    def __init__(self, processedNamesDir_ = os.path.join(PPL_DATA_DIR, 'processed_names')):
        self.processedNamesDir = processedNamesDir_

        self.data = {}

        for filename in glob.glob(os.path.join(processedNamesDir_, '*.json')):
            with open(filename, encoding='utf8') as ifile:
                self.data.update(json.load(ifile))

    def get_occupation(self, name):

        if name in self.data:
            res = self.data[name]['occupation']

        else:
            # hack for the issue of matching utf8 names
            fuzzyMatches = difflib.get_close_matches(name, self.data.keys(), 1)

            if fuzzyMatches:
                fuzzyMatchedName = fuzzyMatches[0]
                print('FUZZY MATCH: %s -> %s' % (name, fuzzyMatchedName))

                res = self.data[fuzzyMatchedName]['occupation']
            else:
                print('ERROR: %s does not exist in names json files. Not even a fuzzy match.' % name)

                res = None

        return res


def file2vec_mass(embeddings_filekey_, outputDir_, occReader_):
    """
    convert all text files in a directory to matrices using a given embedding
    :type occReader_: OccupationReader
    """

    embeddings_filename_ = \
        {'6B50d': '../data/glove/glove.6B/glove.6B.50d.txt',
         '6B300d': '../data/glove/glove.6B/glove.6B.300d.txt',
         '42B300d': '../data/glove/glove.42B.300d.txt',
         '840B300d': '../data/glove/glove.840B.300d.txt',
         'earlylife128d_alltokens': '../data/peopleData/embeddings/earlyLifeEmbeddings.128d_alltokens.txt',
         'earlylife128d_80pc': '../data/peopleData/embeddings/earlyLifeEmbeddings.128d_80pc.txt',
         'earlylife200d_alltokens': '../data/peopleData/embeddings/earlyLifeEmbeddings.200d_alltokens.txt',
         'earlylife200d_80pc': '../data/peopleData/embeddings/earlyLifeEmbeddings.200d_80pc.txt'
         }

    embeddings, _ = extract_embedding(
        embeddingsFilename_=embeddings_filename_[embeddings_filekey_],
        relevantTokens_=None,
        includeUnk_=True
    )
    print('DONE reading embeddings.')

    # read { name: occupation }
    # peopleData = {}
    # for filename in glob.glob(os.path.join(PPL_DATA_DIR, 'processed_names/*.json')):
    #     with open(filename, encoding='utf8') as ifile:
    #         peopleData.update(json.load(ifile))


    processed = 0
    # outputDir = os.path.join(PPL_DATA_DIR, 'earlyLifesWordMats_' + embeddings_filekey_)
    if not os.path.exists(outputDir_): os.mkdir(outputDir_)
    inputFiles = glob.glob(os.path.join(PPL_DATA_DIR, 'earlyLifesTexts/*.txt'))

    for filename in inputFiles:

        name = filename2name(filename)
        outputFname = os.path.join(outputDir_, name + '.json')

        if os.path.exists(outputFname):
            print(outputFname, 'already exists. Skipping...')
            continue

        occupation = occReader_.get_occupation(name)
        mat = file2vec(filename, embeddings, occupation)

        if len(mat)==0:
            print('ERROR: no content read for %s. Skipping...' % name)
            continue

        with open(outputFname, 'w', encoding='utf8') as ofile:
            json.dump({'occupation': occupation, 'mat': [list(l) for l in list(mat)]}, ofile)

        processed += 1

    print('Processed %d out of %d files.' % (processed, len(inputFiles)))



def file2tokens_mass(outputFname_, occupationReader_, selectedOccupations=None):
    """
    For each file in a directory, create a JSON object {'content': cleaned text, 'occupation'}.
    Dump everything into the same file.
    :type occupationReader_: OccupationReader
    """

    def isOccRelevant(occ):
        if selectedOccupations is None: return True

        return (occ[-1] if type(occ)==list else occ) in selectedOccupations

    inputFilenames = glob.glob(os.path.join(PPL_DATA_DIR, 'earlyLifesTexts/*.txt'))
    processed = 0
    res = []

    for filename in inputFilenames:

        name = filename2name(filename)
        occupation = occupationReader_.get_occupation(name)

        if occupation is None or not isOccRelevant(occupation):
            continue

        # clean text
        with open(filename, encoding='utf8') as ifile:
            text = ' '.join(line.strip() for line in ifile.readlines())     # new lines are ignored.
        content = insert_spaces_into_corpus(text)

        if len(content)==0:
            print('ERROR: no content read for %s. Skipping...' % name)
            continue

        res.append({'content': content, 'occupation': occupation, 'name': name})

        processed += 1

    with open(outputFname_, 'w', encoding='utf8') as ofile:
        json.dump(res, ofile)

    print('Processed %d out of %d files.' % (processed, len(inputFilenames)))


if __name__ == '__main__':
    file2tokens_mass(os.path.join(PPL_DATA_DIR, 'earlyLifeTokensFile_polsci.json'),
                     OccupationReader(),
                     ['politician', 'scientist'])