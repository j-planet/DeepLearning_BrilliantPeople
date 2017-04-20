from data_reader import DataReader

DATA_DIRs = {'tiny_fake_2': './data/peopleData/2_samples',
             'tiny_fake_4': './data/peopleData/4_samples',
             'small_2occupations': './data/peopleData/earlyLifesWordMats/politician_scientist',
             'small': './data/peopleData/earlyLifesWordMats',
             'full_2occupations': './data/peopleData/earlyLifesWordMats_42B300d/politician_scientist',
             'full': './data/peopleData/earlyLifesWordMats_42B300d'}

dr = DataReader(DATA_DIRs['small_2occupations'], 'bucketing')

batchSize = 20
numSteps = 100
validEvery = 10

for i in range(numSteps):
    print(i)
    dr.get_next_training_batch(batchSize)

    if i % validEvery == 0:
        dr._get_data_in_batches(20, 'validate')