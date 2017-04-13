import os, json
from pprint import pprint
from data_processing.crawling.utilities import url_2_soup, clean_line



PPL_DATA_DIR = '../../data/peopleData'

for title in ['presidents', 'prime_ministers']:

    url = 'https://en.wikipedia.org/wiki/List_of_current_' + title
    soup = url_2_soup(url)

    res = {}

    for line in soup.select('#mw-content-text td a'):

        line = clean_line(line.text)
        numTokens = len(line.split(' '))

        if numTokens>=2 and numTokens<=4:   # potential name
            res[line] = {'occupation': [title, 'politician']}

    pprint(res)
    print('%d %s names read.' % (len(res), title))

    with open(os.path.join(PPL_DATA_DIR, title + '_processed_names.json'), 'w', encoding='utf8') as outputFile:
        json.dump(res, outputFile)