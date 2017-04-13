import re
from data_processing.file2vec import ZERO_WIDTH_UNICODES
import requests
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
        print('ERROR: no year information available. :', line)
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