import requests
import json
from pprint import pprint

def process_one_name(name):
    name = name.title()
    url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&meta=&titles=' + name.replace(' ', '+') + '&redirects=1'
    t = requests.get(url)
    fullD = json.loads(str(t.content, 'utf-8'))
    extract = list(fullD['query']['pages'].items())[0][1]['extract']

    # pprint(extract, width=200)

    with open('./data/extracts/' + name.replace(' ', '_') + '.txt', 'w', encoding='utf-8') as ofile:
        ofile.writelines(extract)

# process_one_name('albert einstein')

total = 0
processed = 0
failed = 0

for name in json.load(open('./data/processed_names.json')).keys():
    total += 1
    try:
        process_one_name(name)
    except Exception as e:
        print('ERROR for name :', name, e)
        failed += 1
        continue

    processed += 1

print(processed, ' out of ', total, ' processed successfully.')