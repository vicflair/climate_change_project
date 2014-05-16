#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import re
import sys

sys.path.insert(0, 'py-rstr-max/')
from tools_karkkainen_sanders import *
from rstr_max import *

# Get ENB corpus
enb_file = '../enb/ENB_Reports.csv'
enb = []
with open(enb_file, 'r') as f:
    data = csv.reader(f, delimiter='\t')
    enb = [row[7] for row in data]
full_text = ' '.join(enb)

# Find all reports with unicode characters, e.g., \xc2, \x94, etc.
bad = [i for i, e in enumerate(enb) if re.findall('[^\x00-\x7f]', e)]

# Remove terms beginning with [ (as opposed to ![])
sub1 = re.sub('\[[^ ]+', '', full_text)
# Remove terms beginning with http
sub2 = re.sub('http[^ ]+', '', sub1)
# Remove or replace all unicode but not ascii characters
sub3 = re.sub('[^\x00-\x7f]', '', sub2)
# str1 = sub2.decode('ascii', 'ignore')
# str1 = unidecode.unidecode(sub2)
# str1 = sub2.encode('utf-8')
# Finally, compress all spaces
str1 = re.sub('[ ]+', ' ', sub3)
str1_unicode = unicode(str1, 'utf-8', 'replace')

rstr = Rstr_max()
rstr.add_str(str1_unicode)  #str1
r = rstr.go()

print 'done'

def get_repeats(r, min_length, min_count):
    repeats = []
    for (offset_end, nb), (l, start_plage) in r.iteritems():
        ss = rstr.global_suffix[offset_end - l:offset_end]
        id_chaine = rstr.idxString[offset_end - 1]
        s = rstr.array_str[id_chaine]
        if len(ss) > min_length and nb > min_count:
            repeats.append((ss.encode('utf-8'), len(ss), nb))
    return repeats

def sort_repeats(repeats, facet):
    assert type(facet) is int
    ordered = sorted(repeats, key=lambda tup: tup[facet], reverse=True)
    return ordered

repeats = sort_repeats(get_repeats(r, 60, 2), 2)

with open('repeated_maximal_phrases.txt', 'w') as f:
    for repeat in repeats:
        f.write('\"'+repeat[0]+'\"'+', repeated ' + str(repeat[2]) + ' times.\n')