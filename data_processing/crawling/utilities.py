import re
from data_processing.file2vec import ZERO_WIDTH_UNICODES
import requests
from pprint import pprint
import os, json
from bs4 import BeautifulSoup


def parse_name_years(line):
    """
    only 4-digit numbers are considered years
    :param line: sth like Dirck van Baburen (1595–1624), Dutch Carravagist painter 
    :return: (name, birth, death) if no death=None if still alive 
    """

    name = line.split('(')[0]
    rest = ' '.join(line.split('(')[1:])

    years = re.findall(r'\d{4}', rest)

    if len(years)==0:
        birthYear = None
        deathYear = None

    elif len(years)==1:   # no death year
        birthYear = int(years[0])
        deathYear = None

    elif len(years)==2:
        birthYear = int(years[0])
        deathYear = int(years[1])

    else:
        print('ERROR: more than two year numbers available. Took the first 2 and hoped for the best. :', line)
        birthYear = int(years[0])
        deathYear = int(years[1])

    description = ' '.join(rest.split(')')[1:])

    return name.strip(), birthYear, deathYear, description


def clean_line(line_):  # strip and remove stuff, lower case
    res = line_.strip().lower()

    for t in ZERO_WIDTH_UNICODES:
        res.replace(t, ' ')

    return res

def url_2_soup(url):
    return BeautifulSoup(requests.get(url).content, 'html.parser')

def is_not_name(s_):
    """ 
    :return: true if s_ is DEFINITELY NOT a name; false if s_ MAY BE a name  
    """

    if len(s_) < 5: return True

    numTokens = len(s_.split(' '))

    return numTokens < 2 or numTokens > 4 \
           or len(re.findall(r'\d\d+', s_))>0 \
           or s_.find('list of') != -1 \
           or s_.find('u.s.')!=-1 \
           or s_.find(' by ')!=-1

def crawl_wiki_list_of(url, title, occupations, outputDir, verbose=False):
    soup = url_2_soup(url)

    res = {}

    for li in soup.select('#mw-content-text li'):
        line = clean_line(li.text)


        if is_not_name(line): continue  # filter out things most likely not a name

        name, birthYear, deathYear, description = parse_name_years(line)

        res[name] = {'occupation': occupations,
                     'yearBirth': birthYear, 'yearDead': deathYear,
                     'description': title + '.' + description}

    if verbose: pprint(res)
    print('%d %ss names read.' % (len(res), title))

    if outputDir:
        with open(os.path.join(outputDir, title + '_processed_names.json'), 'w', encoding='utf8') as outputFile:
            json.dump(res, outputFile)

    return res