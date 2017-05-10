import json, os
from pprint import pprint

from data_processing.crawling.utilities import crawl_wiki_list_of


PPL_DATA_DIR = '../../data/peopleData'
res = {}

# INPUT_DATA = {
#     'url': 'https://en.wikipedia.org/wiki/List_of_Hollywood_actresses_by_nationality',
#     'occupation': ['actress', 'entertainment'],
#     'descriptionPrepend': 'actress'
#
#     # 'url': 'https://en.wikipedia.org/wiki/List_of_Italian-American_actors',
#     # 'occupation': ['actor', 'entertainment'],
#     # 'descriptionPrepend': 'actor'
#
#     # 'url': 'https://en.wikipedia.org/wiki/List_of_inventors',
#     # 'occupation': ['inventor', 'scientist'],
#     # 'descriptionPrepend': 'inventor'
#
#     # url = 'https://en.wikipedia.org/wiki/List_of_sculptors'
#     # occupation = ['sculptor', 'artist']
#     # 'descriptionPrepend': 'sculptor'
# }

for listname in ['Hollywood_actresses_by_nationality',
                 'American_television_actresses',
                 'American_film_actresses',
                 'Italian_actresses',
                 'Canadian_actors_and_actresses']:

    res.update(crawl_wiki_list_of(
        url = 'https://en.wikipedia.org/wiki/List_of_' + listname,
        title= 'actress or actor',
        occupations= ['actress or actor', 'entertainment'],
        outputDir=None))


for cat in ['American_male_actors']:

    res.update(crawl_wiki_list_of(
        url = 'https://en.wikipedia.org/wiki/Category:' + cat,
        title= 'actor',
        occupations= ['actor', 'entertainment'],
        outputDir=None))


with open('../../data/peopleData/processed_names/actor_processed_names.json', 'w', encoding='utf8') as outputFile:
    json.dump(res, outputFile)