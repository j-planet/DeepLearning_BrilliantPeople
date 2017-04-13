import json, os
import string

from data_processing.crawling.utilities import clean_line, parse_name_years, url_2_soup


PPL_DATA_DIR = '../../data/peopleData'
res = {}

for letter in string.ascii_uppercase:
    print('============== %s ==============' % letter)

    soup = url_2_soup('https://en.wikipedia.org/wiki/List_of_painters_by_name_beginning_with_%22' + letter + '%22')

    for li in soup.select('#mw-content-text li'):
        line = clean_line(li.text)
        name, birthYear, deathYear, description = parse_name_years(line)

        res[name] = {'occupation': ['painter', 'artist'],
                     'yearBirth': birthYear, 'yearDead': deathYear, 'description': 'painter.' + description}

        print(name)

print(len(res), 'painter names read.')
print('done')

with open(os.path.join(PPL_DATA_DIR, 'painters_processed_names.json'), 'w', encoding='utf8') as outputFile:
    json.dump(res, outputFile)