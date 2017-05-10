import json
from data_processing.crawling.utilities import clean_line, parse_name_years, url_2_soup, crawl_wiki_list_of


PPL_DATA_DIR = '../../data/peopleData'

res = {}

for category in ['American_businesspeople',
                 'Manufacturing_company_founders',
                 'Retail_company_founders',
                 'Real_estate_company_founders',
                 'Technology_company_founders',
                 'British_businesspeople',
                 'Canadian_businesspeople',
                 'Japanese_businesspeople',
                 'Businesspeople_awarded_knighthoods',
                 'Industrialists',

                 'Investors',
                 'Private_equity_and_venture_capital_investors',
                 'Women_investors',
                 'Women_business_executives',
                 'Non-profit_executives',
                 'Businesspeople_in_advertising',
                 'Bankers',
                 'Automotive_pioneers',
                 'Businesspeople_in_real_estate',
                 'Businesspeople_in_fashion',
                 'Businesspeople_in_insurance']:

    url = 'https://en.wikipedia.org/wiki/Category:' + category
    res.update(crawl_wiki_list_of(url, 'businessman', ['businessman'], None))


print('Total:', len(res))
with open('../../data/peopleData/processed_names/businessman_processed_names.json', 'w', encoding='utf8') as outputFile:
    json.dump(res, outputFile)