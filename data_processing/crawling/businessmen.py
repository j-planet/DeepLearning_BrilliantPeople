import json
from data_processing.crawling.utilities import clean_line, parse_name_years, url_2_soup, crawl_wiki_list_of


PPL_DATA_DIR = '../../data/peopleData'

res = {}

for category in ['American_businesspeople',
                 'Manufacturing_company_founders',
                 'Retail_company_founders',
                 'Real_estate_company_founders',
                 'Technology_company_founders']:

    url = 'https://en.wikipedia.org/wiki/Category:' + category
    res.update(crawl_wiki_list_of(url, 'businessman', ['businessman'], None))


print('Total:', len(res))
with open('../../data/peopleData/processed_names/businessman_processed_names.json', 'w', encoding='utf8') as outputFile:
    json.dump(res, outputFile)