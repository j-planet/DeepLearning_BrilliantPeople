import json
from data_processing.crawling.utilities import clean_line, parse_name_years, url_2_soup, crawl_wiki_list_of


PPL_DATA_DIR = '../../data/peopleData'

res = {}

for category in ['Leaders_of_political_parties',
                 'Politicians_awarded_knighthoods',
                 'Politicians_with_physical_disabilities',
                 'Russian_politicians',
                 'Austrian_politicians',
                 'Albanian_politicians',
                 'German_politicians',
                 'French_politicians',
                 'Turkish_politicians',
                 'Polish_politicians',
                 'Hungarian_politicians',

                 'Chinese_politician_stubs',
                 'Chinese_politicians_convicted_of_crimes',
                 'North_Korean_politicians',
                 'Norwegian_politicians',
                 'Belgian_politicians']:

    url = 'https://en.wikipedia.org/wiki/Category:' + category
    res.update(crawl_wiki_list_of(url, 'politician', ['politician'], None))


print('Total:', len(res))
with open('../../data/peopleData/processed_names/politicians_processed_names.json', 'w', encoding='utf8') as outputFile:
    json.dump(res, outputFile)