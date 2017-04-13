import requests
import json
import os

PPL_DATA_DIR = '../../data/peopleData'

def possible_spellings_of_a_name(name_):

    name_ = name_.strip()

    return [name_.lower(), name_.title(), name_.replace('.', ''), name_.replace('i', 'I'),
            name_.split()[0] + ' ' + ' '.join(name_.split()[1:]).title()  # 14th Dalai Lama
            ]


def crawl_wiki_text(name_, outputDir_, skipIfExists_):
    """ 
    :return: if successful: (extract, output filename); otherwise, None
    """

    outputFilename = os.path.join(outputDir_, name_ + '.txt')

    if skipIfExists_ and os.path.exists(outputFilename):
        print(name_, 'already exists. Skipping...')

        with open(outputFilename, encoding='utf8') as file:
            existingExtract = file.readlines()

        return existingExtract, outputFilename

    # try multiple spellings of the name
    for curName in possible_spellings_of_a_name(name_):

        url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&meta=&titles=' + curName.replace(' ', '+') + '&redirects=1'
        t = requests.get(url)
        fullD = json.loads(str(t.content, 'utf-8'))

        extract = list(fullD['query']['pages'].items())[0][1].get('extract', None)

        if extract is not None:

            with open(outputFilename, 'w', encoding='utf-8') as ofile:
                ofile.writelines(extract)

            return extract, outputFilename

    print('extract does not exist for %s. quitting...' % name_)
    return None


if __name__ == '__main__':

    inputFileName = os.path.join(PPL_DATA_DIR, 'nobel_prize_processed_names.json')
    outputDir = os.path.join(PPL_DATA_DIR, 'extracts', 'nobelprizeextracts')
    skipIfFileExists = True

    total = 0
    processed = 0
    failed = 0

    with open(inputFileName) as namesFiles:

        for name in json.load(namesFiles).keys():

            print(total, name)
            total += 1

            if crawl_wiki_text(name, outputDir, skipIfFileExists) is not None:
                processed += 1


    print(processed, ' out of ', total, ' processed successfully.')