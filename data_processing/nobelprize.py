from pprint import pprint
import numpy as np
import collections
import json
import os

from file2vec import ZERO_WIDTH_UNICODES


PRIZE_STARTING_STRS = ['The Nobel Prize in'.lower(),
                       'The Nobel Peace Prize'.lower(),
                       'The Sveriges Riksbank Prize in Economic Sciences'.lower()]

NO_PRIZE_STR = 'No Nobel Prize'.lower()

PRIZE_TO_OCCUPATIONS = {'The Nobel Peace Prize'.lower(): ['social'],
                        'The Nobel Prize in Chemistry'.lower(): ['chemist', 'scientist'],
                        'The Nobel Prize in Literature'.lower(): ['author'],
                        'The Nobel Prize in Physics'.lower(): ['physicist', 'scientist'],
                        'The Nobel Prize in Physiology or Medicine'.lower(): ['physiology or medicine','scientist'],
                        'The Sveriges Riksbank Prize in Economic Sciences'.lower(): ['economist', 'scientist']}

PPL_DATA_DIR = '../data/peopleData'

res = []
prizeName = None
namesNDescriptions = []

nameInNextLine = False
descriptionInNextLine = False

def clean_line(line_):  # strip and remove stuff, lower case
    res = line_.strip().lower()

    for t in ZERO_WIDTH_UNICODES:
        res.replace(t, ' ')

    return res

def store_previous():
    global prizeName, namesNDescriptions

    # print('--- wrapping up previous:', prizeName)
    # pprint(namesNDescriptions)
    # print('\n')

    if prizeName is not None:
        res.append((prizeName, namesNDescriptions))

    namesNDescriptions = []
    prizeName = None

def parse_names(namesStr_):
    return [name.strip() for name in namesStr_.replace(' and ', ',').split(',')]

def parse_occupation(prizeName_):
    for k, v in PRIZE_TO_OCCUPATIONS.items():
        if prizeName_.lower().find(k) == 0:
            return v

    print('ERROR! No occupations found for prize:', prizeName_)


inputFile = open(os.path.join(PPL_DATA_DIR, 'nobelprize.txt'), encoding='utf8')

for line in inputFile.readlines():

    line = clean_line(line)

    if line == '':

        descriptionInNextLine = False
        continue

    if line.isnumeric():  # year number. e.g. 2013

        store_previous()

        nameInNextLine = False
        descriptionInNextLine = False
        continue

    if nameInNextLine:

        if line.find(NO_PRIZE_STR)==0:
            prizeName = None
            namesNDescriptions = []
        else:
            namesNDescriptions.append({'names': line})
            descriptionInNextLine = True

        nameInNextLine = False
        continue

    if descriptionInNextLine:
        namesNDescriptions[-1]['description'] = line

        descriptionInNextLine = False
        continue

    if np.any([line.find(s)==0 for s in PRIZE_STARTING_STRS]):

        store_previous()
        prizeName = line
        nameInNextLine = True
        descriptionInNextLine = False

        continue

    if prizeName is not None:   # multiple people for the same award
        names = line

        descriptionInNextLine = True
        nameInNextLine = False
        continue

    print('ERROR! Dunno how to categorize line:', line)

inputFile.close()


# ------------ format crawled stuff ------------
formattedRes = {}

for prizeName, namesNDescriptions in res:

    occupations = parse_occupation(prizeName)

    for nameNDescriptionDict in namesNDescriptions:

        names = parse_names(nameNDescriptionDict['names'])
        desc = nameNDescriptionDict.get('description', prizeName)

        for name in names:
            if name in formattedRes:
                print('Multiple prize for the same person: %s, previous occupation: %s, current prize: %s' %(name, formattedRes[name]['occupation'][-1], prizeName))
                if occupations[-1] != formattedRes[name]['occupation'][-1]:
                    print('Reluctantly skipping the incredible %s for "conflict" in occupation.' % name)
            else:
                formattedRes[name] = {'occupation': occupations, 'description': desc}

print('------------ format crawled stuff ------------')
pprint(formattedRes)

print('\n------------ Summary ------------')
pprint(collections.Counter([v['occupation'][-1] for v in formattedRes.values()]))

with open(os.path.join(PPL_DATA_DIR, 'nobel_prize_processed_names.json'), 'w', encoding='utf8') as outputFile:
    json.dump(formattedRes, outputFile)