import collections

with open('./data/text8.txt', 'r') as ifile:
    words = ifile.read().split()
    print('%d non-unique words.' % len(words))

tokensByFrequency = collections.Counter(words).most_common()
print('%d unique words.' % len(tokensByFrequency))

with open('./data/text8_sorted_by_frequency.txt', 'w') as ofile:
    for token, count in tokensByFrequency:
        ofile.write('%s %d\n' % (token, count))