import json, os
from pprint import pprint

from data_processing.crawling.utilities import crawl_wiki_list_of


PPL_DATA_DIR = '../../data/peopleData'

INPUT_DATA = {
    'url': 'https://en.wikipedia.org/wiki/List_of_Hollywood_actresses_by_nationality',
    'occupation': ['actress', 'entertainment'],
    'descriptionPrepend': 'actress'

    # 'url': 'https://en.wikipedia.org/wiki/List_of_Italian-American_actors',
    # 'occupation': ['actor', 'entertainment'],
    # 'descriptionPrepend': 'actor'

    # 'url': 'https://en.wikipedia.org/wiki/List_of_inventors',
    # 'occupation': ['inventor', 'scientist'],
    # 'descriptionPrepend': 'inventor'

    # url = 'https://en.wikipedia.org/wiki/List_of_sculptors'
    # occupation = ['sculptor', 'artist']
    # 'descriptionPrepend': 'sculptor'
}

crawl_wiki_list_of(url = 'https://en.wikipedia.org/wiki/List_of_Hollywood_actresses_by_nationality',
                   title= 'actress',
                   occupations= ['actress', 'entertainment'],
                   outputDir=PPL_DATA_DIR)