import json, os

from data_processing.crawling.utilities import clean_line, parse_name_years, url_2_soup


PPL_DATA_DIR = '../../data/peopleData'
soup = url_2_soup('https://en.wikipedia.org/wiki/List_of_sculptors')

res = {}

for li in soup.select('#mw-content-text li'):
    line = clean_line(li.text)

    # most likely not a name
    if len(line) < 5: continue
    if line.find('list of') != -1: continue

    name, birthYear, deathYear, description = parse_name_years(line)

    res[name] = {'occupation': ['sculptor', 'artist'],
                 'yearBirth': birthYear, 'yearDead': deathYear, 'description': 'sculptor.' + description}


print(len(res), 'sculptors names read.')
print('done')

with open(os.path.join(PPL_DATA_DIR, 'sculptors_processed_names.json'), 'w', encoding='utf8') as outputFile:
    json.dump(res, outputFile)