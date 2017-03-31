import glob
import string

from contexts import read_people_names, read_occupations


def remove_punctuations(s):
    return s.translate(str.maketrans({key: ' ' for key in string.punctuation + '\n'}))


def create_early_life_corpus(filenames, outputFname, removePunctuations):

    res = ''
    numSkipped = 0

    for file in filenames:
        try:
            with open(file, encoding='utf8') as ifile:
                text = ' '.join(remove_punctuations(ifile.read()).lower().split()) \
                    if removePunctuations else ' '.join(ifile.read().lower().split())
        except Exception as e:
            print(e)
            numSkipped += 1
            continue

        res += text + ' '

    if outputFname:
        with open(outputFname, 'w', encoding='utf8') as ofile:
            ofile.write(res)

    print('%d out of %d skipped.' %(numSkipped, len(filenames)))
    return res


def create_early_life_corpus_by_occupation(occupation, outputFname, removePunctuations):

    filenames = []

    for name in read_people_names(occupation=occupation):

        convertedName = '_'.join([t.capitalize() for t in name.split()])
        filenames.append('./data/peopleData/earlyLifes/' + convertedName + '.txt')

    return create_early_life_corpus(filenames, outputFname, removePunctuations)


def create_early_life_corpus_for_all(outputFname, removePunctuations):
    return create_early_life_corpus(
        glob.glob('./data/peopleData/earlyLifes/*.txt'),
        outputFname, removePunctuations
    )


if __name__ == '__main__':
    create_early_life_corpus_for_all('./data/peopleData/earlyLifeCorpus.txt', removePunctuations=False)

    # for occupation in read_occupations():
    #     print('\n========', occupation)
    #     create_early_life_corpus_by_occupation(occupation,
    #                                            outputFname='./data/peopleData/earlyLifeCorpus_' + occupation + '.txt')