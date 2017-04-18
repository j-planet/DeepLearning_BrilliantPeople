import json
import os, glob
import shutil



PEOPLE_DATA_DIR = '../data/peopleData'


def makeSmallSamples(occupations_, srcDir, overwriteIfExists=False):
    outputDir = os.path.join(srcDir, '_'.join(occupations_))

    if os.path.exists(outputDir):
        print('Output directory %s already exists. %s...' % (outputDir, 'Overwriting' if overwriteIfExists else 'skipping'))

        if not overwriteIfExists:
            return
    else:
        os.mkdir(outputDir)

    for filePath in glob.glob(os.path.join(srcDir, '*.json')):

        with open(filePath, encoding='utf8') as inputFile:
            d = json.load(inputFile)
            
        occupation = d['occupation'][-1] if type(d['occupation'])==list else d['occupation']

        if occupation in occupations_:

            print('Copying', filePath)
            shutil.copyfile(filePath, os.path.join(outputDir, os.path.basename(filePath)))


if __name__ == '__main__':

    occupations = ['politician', 'scientist']

    makeSmallSamples(occupations, srcDir=os.path.join(PEOPLE_DATA_DIR, 'earlyLifesWordMats_42B300d'),overwriteIfExists=True)