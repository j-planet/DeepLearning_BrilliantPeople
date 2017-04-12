from pprint import pprint
import numpy as np

PRIZE_STARTING_STRS = ['The Nobel Prize in'.lower(),
                       'The Nobel Peace Prize'.lower(),
                       'The Sveriges Riksbank Prize in Economic Sciences'.lower()]

NO_PRIZE_STR = 'No Nobel Prize'.lower()

res = []
prizeName = None
namesNDescriptions = []

nameInNextLine = False
descriptionInNextLine = False

def store_previous():
    global prizeName, namesNDescriptions

    print('--- wrapping up previous:', prizeName)
    pprint(namesNDescriptions)
    print('\n')

    if prizeName is not None:
        res.append((prizeName, namesNDescriptions))

    namesNDescriptions = []
    prizeName = None



inputFile = open('../data/peopleData/nobelprize.txt', encoding='utf8')

for line in inputFile.readlines():

    line = line.strip().lower()

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

    if prizeName is not None:
        names = line

        descriptionInNextLine = True
        nameInNextLine = False
        continue

    print('ERROR! Dunno how to categorize line:', line)

inputFile.close()

