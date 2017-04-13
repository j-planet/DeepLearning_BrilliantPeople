from pprint import pprint
import json
import os

from data_processing.crawling.utilities import clean_line, parse_name_years


PPL_DATA_DIR = '../../data/peopleData'
res = {}



if __name__ == '__main__':

    with open(os.path.join(PPL_DATA_DIR, 'mathies.txt'), encoding='utf8') as namesFile:

        for line in namesFile.readlines():

            line = clean_line(line)

            name, birthYear, DeathYear, _ = parse_name_years(line)

            res[name] = {'occupation': ['mathematician', 'scientist'],
                         'yearBirth': birthYear, 'yearDead': DeathYear, 'description': '19th century mathematician'
                         }

    pprint(res)
    print(len(res), '19th-century mathematicians.')
    with open(os.path.join(PPL_DATA_DIR, 'mathies_processed_names.json'), 'w', encoding='utf8') as outputFile:
        json.dump(res, outputFile)