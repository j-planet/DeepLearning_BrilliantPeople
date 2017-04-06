import json
import os
import shutil



PEOPLE_DATA_DIR = '../data/peopleData'


def makeSmallSamples(occupations):
    outputDir = os.path.join(PEOPLE_DATA_DIR, 'earlyLifesWordMats', '_'.join(occupations))

    if os.path.exists(outputDir):
        print('Output directory %s already exists. Aborting...' % outputDir)
        return

    os.mkdir(outputDir)
    with open(os.path.join(PEOPLE_DATA_DIR, 'processed_names.json'), encoding='utf8') as inputFile:
        peopleData = json.load(inputFile)
    for name, d in peopleData.items():

        if d['occupation'][-1] in occupations:

            srcFname = os.path.join(PEOPLE_DATA_DIR, 'earlyLifesWordMats', name + '.json')

            if os.path.exists(srcFname):
                print('Copying', name)
                targetFname = os.path.join(outputDir, name + '.json')
                shutil.copyfile(srcFname, targetFname)


if __name__ == '__main__':

    occupations = ['politician', 'scientist']

    makeSmallSamples(occupations)