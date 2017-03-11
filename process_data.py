#!/usr/bin/python3

import re
from pprint import pprint
import json

res = {}    # {name: {yearBirth: ..., yearDead: ..., description: ...}}

# ================== processing names1.txt ==================
def process_names1():
    file = open('./data/names1.txt', errors='strict')

    for line in file.readlines():

        try:
            line = line.strip()
            if line == '': continue

            name = re.split("[(-]", line)[0].strip().lower()
            description = line.split(')')[1].strip().lstrip('- ')

            yearStr = line.split('(')[1].split(')')[0].strip()
            yearBirth = re.split("[- ]", yearStr)[0].strip()
            yearDead = re.split("[- ]", yearStr)[-1].strip()


            print(line)
            print(name)
            print(yearStr, yearBirth, yearDead)
            print(description)

            if name in res:
                print(name, 'already exists. skipping...')
                continue

            res[name] = {'yearBirth': yearBirth, 'yearDead': yearDead, 'description': description}

        except Exception as e:
            print('ERROR:', e)
            print(line)

        print('-' * 10)

    file.close()


# ================== processing names2.txt ==================
def process_names2():

    file = open('./data/names2.txt', errors='strict')

    for line in file.readlines():
        try:
            line = line.strip()
            if line == '': continue

            name, description = line.split(' - ')
            name = name.strip().lower()
            description = description.strip()

            print(line)
            print(name)
            print(description)

            if name in res:
                print(name, 'already exists. skipping...')
                continue

            res[name] = {'description': description}

        except Exception as e:
            print('ERROR:', e)
            print(line)

        print('-' * 10)



    file.close()

process_names1()
process_names2()
pprint(res)
print(len(res)) # ~231 people

with open('./data/processed_names.json', 'w') as ofile:
    json.dump(res, ofile)