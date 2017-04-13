import json
import glob
import os
import bs4
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
import multiprocessing


from data_processing.crawling.crawl_wiki import crawl_wiki_text


# in order of search
EARLY_LIFE_STRINGS = ['early years', 'early life', 'youth', 'childhood', 'personal life', 'birth and family', 'origin']

PPL_DATA_DIR = '../../data/peopleData'


def get_headings(filename):
    file = open(filename, encoding='UTF-8')
    soup = BeautifulSoup(file, 'html.parser', from_encoding='UTF-8')

    for i, element in enumerate(soup):
        if element.name and element.name[0]=='h':
            print(i, element.contents[0].string)

def get_content(element):
    if isinstance(element, bs4.element.NavigableString): return element

    return ''.join(get_content(c) for c in element.contents)

# for elements of h1, h2, ...
def is_smaller_header(large, small):

    largeName = large.name
    smallName = small.name

    if largeName[0] != 'h': return False

    if smallName is None or smallName[0] != 'h': return True

    if len(largeName) != 2 or len(smallName) != 2: print('Do not know how to compare names', largeName, 'and', smallName)

    if largeName[1] < smallName[1]: return True


def extract_early_life(filename_, outputFilename_, skipIfExists_):

    if skipIfExists_ and os.path.exists(outputFilename_):
        print(outputFilename_, 'already exists. Skipping...')

        with open(outputFilename_, encoding='utf8') as file:
            res = file.readlines()
        return res

    # ------- output file does not exist ---------
    with open(filename_, encoding='UTF-8') as inputFile:
        soup = BeautifulSoup(inputFile, 'html.parser', from_encoding='UTF-8')

    earlyLifeContent = []

    for element in soup:

        if earlyLifeContent: break

        if element.name and element.name[0] == 'h':
            heading = get_content(element).lower()

            for earlyLifeString in EARLY_LIFE_STRINGS:

                if earlyLifeString.lower() in heading and heading.find('career')==-1:
                    print('found for %s:' % filename_, earlyLifeString, 'in', heading)

                    for nextSibling in element.next_siblings:

                        if not is_smaller_header(element, nextSibling): break

                        curContent = get_content(nextSibling).strip()

                        if not curContent: continue

                        # TODO: should headers be treated differently? e.g. given more weight, ignored altogether, etc?
                        # start a new "paragraph" if it's a p or blockquote
                        if nextSibling.name in ['p', 'blockquote'] or nextSibling.name[0] == 'h' or len(
                                earlyLifeContent) == 0:
                            earlyLifeContent.append(curContent)
                        else:
                            earlyLifeContent[-1] += curContent

                    with open(outputFilename_, 'w', encoding='utf-8') as outputFile:
                        outputFile.writelines('\n'.join(earlyLifeContent))

                    break

    if not earlyLifeContent:
        print(filename_, 'does not have early life.')

    return earlyLifeContent


def name_to_earlylife(name,
                      extractsDir_ = os.path.join(PPL_DATA_DIR, 'extracts'),
                      outputDir_ = os.path.join(PPL_DATA_DIR, 'earlyLifesTexts'),
                      skipIfFilesExists_=True):

    finalOutputFname = os.path.join(outputDir_, name + '.txt')

    if skipIfFilesExists_ and os.path.exists(finalOutputFname):
        print('Early life for %s already exists. Skipping...' % name)
        return finalOutputFname

    temp = crawl_wiki_text(name, extractsDir_, skipIfFilesExists_)

    if temp is None:
        return None

    return extract_early_life(temp[1], finalOutputFname, skipIfFilesExists_)


if __name__ == '__main__':


    # inputDir = os.path.join(PPL_DATA_DIR, 'extracts')
    # outputDir = os.path.join(PPL_DATA_DIR, 'earlyLifesTexts')
    #
    # skipIfFileExists = True
    #
    # processed = 0
    #
    # for wikiTextFilename in glob.glob(os.path.join(inputDir, '*.txt')):
    #
    #     total += 1
    #
    #     if extract_early_life(wikiTextFilename, os.path.join(outputDir, wikiTextFilename.split('/')[-1])):
    #         processed += 1
    #

    # read { name: occupation }
    peopleData = {}
    for filename in glob.glob(os.path.join(PPL_DATA_DIR, 'processed_names/*processed_names*.json')):
        with open(filename, encoding='utf8') as ifile:
            peopleData.update(json.load(ifile))

    def process_one_person(name_):
        try:
            name_to_earlylife(name_)
        except Exception as e:
            print('Error occurred:', e)


    Parallel(n_jobs=8)(delayed(process_one_person)(name) for name in peopleData.keys())


