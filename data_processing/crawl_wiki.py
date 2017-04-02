import requests
import json
from pprint import pprint


def possible_spellings_of_a_name(name_):

    name_ = name_.strip()

    return [name_.lower(), name_.title(), name_.replace('.', ''), name_.replace('i', 'I'),
            name.split()[0] + ' ' + ' '.join(name.split()[1:]).title()  # 14th Dalai Lama
            ]


def process_one_name(name_):

    # try multiple spellings of the name
    for curName in possible_spellings_of_a_name(name_):

        url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&meta=&titles=' + curName.replace(' ', '+') + '&redirects=1'
        t = requests.get(url)
        fullD = json.loads(str(t.content, 'utf-8'))

        extract = list(fullD['query']['pages'].items())[0][1].get('extract', None)

        if extract is not None:

            with open('../data/peopleData/extracts/' + name_ + '.txt', 'w', encoding='utf-8') as ofile:
                ofile.writelines(extract)

            return extract

    print('extract does not exist for %s. quitting...' % name_)
    return None




# process_one_name('albert einstein')

total = 0
processed = 0
failed = 0

for name in json.load(open('../data/peopleData/processed_names.json')).keys():
    total += 1

    if process_one_name(name) is not None:
        processed += 1

print(processed, ' out of ', total, ' processed successfully.')