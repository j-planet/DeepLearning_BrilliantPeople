from pprint import pprint
from data_processing.crawling.utilities import url_2_soup, clean_line, is_not_name
import os, json

res = {}

def _is_not_name(s_):
    s_ = s_.lower()

    return is_not_name(s_) or s_.find('olympic')!=-1 or s_.find('by')!=-1

def part1():
    soup = url_2_soup('http://www.espn.com/espn/feature/story/_/id/15685581/espn-world-fame-100')


    for t in soup.select('h4'):
        line = clean_line(t.text)
        name = ' '.join(line.split('. ')[1:])

        res[name] = {'occupation': ['athlete'], 'description': 'ESPN famous 100'}


def part2():
    for gender in ['men', 'women']:
        soup = url_2_soup('https://en.wikipedia.org/wiki/List_of_Olympic_medalists_in_athletics_(%s)' % gender)

        for t in soup.select('#mw-content-text td a'):
            line = clean_line(t.text)
            if _is_not_name(line): continue

            name = line.split('(')[0]
            res[name] = {'occupation': ['olympian', 'athlete'], 'description': 'olympic medalist'}


part1()
part2()
print('%d athletes read.' % len(res))

with open('../../data/peopleData/processed_names/athlete_processed_names.json', 'w', encoding='utf8') as outputFile:
    json.dump(res, outputFile)