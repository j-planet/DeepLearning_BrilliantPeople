import glob
import string

from blah import read_people_names, read_occupations


def remove_punctuations(s):
    return s.translate(str.maketrans({key: ' ' for key in string.punctuation + '\n'}))


def create_early_life_corpus(filenames, outputFname):

    res = ''
    numSkipped = 0

    for file in filenames:
        try:
            with open(file, encoding='utf8') as ifile:
                text = ' '.join(remove_punctuations(ifile.read()).lower().split())
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


def create_early_life_corpus_by_occupation(occupation, outputFname):

    filenames = []

    for name in read_people_names(occupation=occupation):

        convertedName = '_'.join([t.capitalize() for t in name.split()])
        filenames.append('./data/peopleData/earlyLifes/' + convertedName + '.txt')

    return create_early_life_corpus(filenames, outputFname)


def create_early_life_corpus_for_all(outputFname):
    return create_early_life_corpus(
        glob.glob('./data/peopleData/earlyLifes/*.txt'),
        outputFname
    )


if __name__ == '__main__':
    # create_early_life_corpus_for_all('./data/peopleData/earlyLifeCorpus.txt')

    for occupation in read_occupations():
        print('\n========', occupation)
        create_early_life_corpus_by_occupation(occupation,
                                               outputFname='./data/peopleData/earlyLifeCorpus_' + occupation + '.txt')