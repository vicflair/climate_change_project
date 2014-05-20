g#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Victor Ma
#
# This code package uses the py-rstr-max program to perform repeated phrase
# analysis on the ENB corpus.
# Refer to: https://code.google.com/p/py-rstr-max/
import csv
import re
import sys

# Using the
sys.path.insert(0, 'py-rstr-max/')
from tools_karkkainen_sanders import *
from rstr_max import *


def analyze_repeats():
    # Get ENB corpus
    enb_file = '../enb/ENB_Reports.csv'
    with open(enb_file, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        enb = [row[7] for row in data]
    full_text = ' '.join(enb)
    # Sanitize corpus
    str1 = sanitize(full_text)
    # Use py-rstr-max on sanitized corpus
    str1_unicode = unicode(str1, 'utf-8', 'replace')
    rstr = Rstr_max()
    rstr.add_str(str1_unicode)  #str1
    r = rstr.go()
    return r

def sanitize(text):
    """ Sanitizes the corpus before repeated phrase analysis. Assumes that
    the supplied corpus is a single string.
    """
    # Find all reports with unicode characters, e.g., \xc2, \x94, etc.
    bad = [i for i, e in enumerate(text) if re.findall('[^\x00-\x7f]', e)]
    # Remove terms beginning with [ (as opposed to ![])
    sub1 = re.sub('\[[^ ]+', '', full_text)
    # Remove terms beginning with http
    sub2 = re.sub('http[^ ]+', '', sub1)
    # Remove or replace all unicode but not ascii characters
    sub3 = re.sub('[^\x00-\x7f]', '', sub2)
    # Finally, compress all spaces
    sanitized_text = re.sub('[ ]+', ' ', sub3)
    return sanitized_text


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


def show_phrases(r, min_length=60, min_reps=2, fname=''):
    phrases = sort_repeats(get_repeats(r, min_length, 2), min_reps)
    if fname is not '':
        with open(fname, 'w') as f:
            for phrase in phrases:
                line = ('\"'+phrase[0]+'\"' + ', repeated ' + str(phrase[2]) +
                        ' times.\n')
                f.write(line)
        print 'Wrote maximal repeats to '
    for phrase in phrases:
        print phrase[0], str(phrase[1]), str(phrase[2])
    return phrases