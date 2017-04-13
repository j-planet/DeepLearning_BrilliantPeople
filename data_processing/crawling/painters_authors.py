import json, os
import string

from data_processing.crawling.utilities import clean_line, parse_name_years, url_2_soup, crawl_wiki_list_of


PPL_DATA_DIR = '../../data/peopleData'

urls = ['https://en.wikipedia.org/wiki/List_of_authors_by_name:_' + letter for letter in string.ascii_uppercase]
title = 'author'
occupations = ['author']

# urls = ['https://en.wikipedia.org/wiki/List_of_painters_by_name_beginning_with_%22' + letter + '%22' for letter in string.ascii_uppercase]
# title = 'painter'
# occupations = ['painter', 'artist']


res = {}

for url in urls:
    print('======== %s ========')
    crawledDict = crawl_wiki_list_of(url, title, occupations, outputDir=None)

    res.update(crawledDict)

    # soup = url_2_soup(url)
    #
    # for li in soup.select('#mw-content-text li'):
    #     line = clean_line(li.text)
    #     name, birthYear, deathYear, description = parse_name_years(line)
    #
    #     res[name] = {'occupation': ['painter', 'artist'],
    #                  'yearBirth': birthYear, 'yearDead': deathYear, 'description': 'painter.' + description}
    #
    #     print(name)

print('%d %ss names read.' % (len(res), title))

with open(os.path.join(PPL_DATA_DIR, title + '_processed_names.json'), 'w', encoding='utf8') as outputFile:
    json.dump(res, outputFile)