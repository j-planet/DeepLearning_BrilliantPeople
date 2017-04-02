import json
import glob
import bs4
from bs4 import BeautifulSoup


def get_headings(filename):
    file = open(filename, encoding='UTF-8')
    soup = BeautifulSoup(file, 'html.parser', from_encoding='UTF-8')

    for i, element in enumerate(soup):
        if element.name and element.name[0]=='h':
            print(i, element.contents[0].string)

def get_content(element):
    if isinstance(element, bs4.element.NavigableString): return element

    return ''.join(get_content(c) for c in element.contents)

# for elements of h1, h2, ...
def is_smaller_header(large, small):

    largeName = large.name
    smallName = small.name

    if largeName[0] != 'h': return False

    if smallName is None or smallName[0] != 'h': return True

    if len(largeName) != 2 or len(smallName) != 2: print('Do not know how to compare names', largeName, 'and', smallName)

    if largeName[1] < smallName[1]: return True

# in order of search
EARLY_LIFE_STRINGS = ['early years', 'early life', 'youth', 'childhood', 'personal life', 'life',
                      'birth and family', 'biography', 'origin']


total = 0
processed = 0
failed = 0

for filename in glob.glob('../data/peopleData/extracts/*.txt'):

    print('----', filename)
    file = open(filename, encoding='UTF-8')
    soup = BeautifulSoup(file, 'html.parser', from_encoding='UTF-8')

    earlyLifeContent = []

    for element in soup:

        if earlyLifeContent: break

        if element.name and element.name[0] == 'h':
            heading = get_content(element)

            for earlyLifeString in EARLY_LIFE_STRINGS:
                if earlyLifeString.lower() in heading.lower():
                    print('found:', earlyLifeString, 'in', heading)

                    for nextSibling in element.next_siblings:

                        if not is_smaller_header(element, nextSibling): break

                        curContent = get_content(nextSibling).strip()

                        if not curContent: continue

                        # TODO: should headers be treated differently? e.g. given more weight, ignored altogether, etc?
                        # start a new "paragraph" if it's a p or blockquote
                        if nextSibling.name in ['p', 'blockquote'] or nextSibling.name[0]=='h' or len(earlyLifeContent) == 0:
                            earlyLifeContent.append(curContent)
                        else:
                            earlyLifeContent[-1] += curContent

                    with open('../data/peopleData/earlyLifes/' + filename.split('/')[-1], 'w', encoding='utf-8') as outputFile:
                        outputFile.writelines('\n'.join(earlyLifeContent))

                    break


    if not earlyLifeContent:
        print(filename, 'does not have early life.')

    file.close()