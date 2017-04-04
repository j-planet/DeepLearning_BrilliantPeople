# TODO:
# 1 token -> 1 vector (i.e. sequence)
# bucket corpuses of similar lengths
# train for occupation!
# Then perhaps go 4-dimensional to put sentences together.

import json
from pprint import pprint

with open('./data/peopleData/processed_names.json', encoding='utf8') as ifile:
    data = json.load(ifile)

