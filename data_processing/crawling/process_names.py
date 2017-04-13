#!/usr/bin/python3

import re
import numpy as np
from pprint import pprint
import json

res = {}    # {name: {yearBirth: ..., yearDead: ..., description: ...}}


# occupation: a list [most specific, ..., most generic]. E.g. [president, politician]
def get_occuptation(description, name):

    des = description.lower()
    name = name.lower()

    if np.any([keyword in des for keyword in ['runner', 'football', 'basketball', 'baseball', 'tennis', 'golf', 'cyclist', 'athlete']]):
        return ['athlete']

    if np.any([keyword in des for keyword in ['actress', 'singer', 'model', 'actor', 'film']]):
        return ['entertainment']

    if np.any([keyword in des for keyword in ['musician', 'composer']]):
        return ['musician', 'artist']

    if 'fashion' in des:
        return ['fashion', 'artist']

    if 'artist' in des:
        return ['artist']

    if 'president' in des:
        return ['president', 'politician']

    if 'queen' in des or 'queen' in name:
        return ['queen', 'royalty']

    if 'king' in des or 'king' in name:
        return ['king', 'royalty']

    if 'emperor' in des or 'emperor' in name:
        return ['emperor', 'royalty']

    if np.any([keyword in des for keyword in ['monarch', 'throne']]):
        return ['royalty']

    if 'pope' in des or 'pope' in name:
        return ['pope', 'religion']


    if np.any([keyword in des for keyword in ['politic', 'prime minister', 'chancellor']]):
        return ['politician']

    if  np.any([keyword in des for keyword in ['christian', 'missionary', 'bishop', 'jesus', 'god', 'religion', 'pastor', 'priest']]):
        return ['religion']

    if np.any([keyword in des for keyword in ['humanitarian', 'civil rights', 'movement', 'rights', 'nationalist', 'social issue']]):
        return ['social']

    if np.any([keyword in des for keyword in ['author', 'poet', 'writer']]):
        return ['author']

    if np.any([keyword in des for keyword in
               ['physic', 'chemistry', 'biolog', 'math', 'scientist', 'philosopher', 'economist', 'inventor']]):
        return ['scientist']

    if np.any([keyword in des for keyword in ['entrepreneur', 'businessman', 'industrialist']]):
        return ['businessman']

    if np.any([keyword in des for keyword in
               ['explorer', 'astronaut', 'aviator']]):
        return ['explorer']

    return input('Description: %s. ---> Enter occupation:  ' % des).split(',')


# ================== processing names1.txt ==================
def process_names1():
    file = open('./data/peopleData/names1.txt', errors='strict')

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

            res[name] = {
                'yearBirth': yearBirth, 'yearDead': yearDead, 'description': description,
                'occupation': get_occuptation(description, name)
            }

        except Exception as e:
            print('ERROR:', e)
            print(line)

        print('-' * 10)

    file.close()


# ================== processing names2.txt ==================
def process_names2():

    file = open('./data/peopleData/names2.txt', errors='strict')

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

            res[name] = {
                'description': description,
                'occupation': get_occuptation(description, name)
            }

        except Exception as e:
            print('ERROR:', e)
            print(line)

        print('-' * 10)



    file.close()

process_names1()
process_names2()
pprint(res)
print(len(res)) # ~231 people

with open('./data/peopleData/processed_names.json', 'w') as ofile:
    json.dump(res, ofile)