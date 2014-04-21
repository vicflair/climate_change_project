import numpy
import tokenize
import re

fname = 'enb_corpus_kpex.kpex_n9999.txt'
with open(fname, 'r') as f:
    data = f.readlines()

for i in range(0, 10):
    line = data[i]
    with_synonym = re.search('([^:]*)::SYN::([\w]*),F ([\d]*)', line)
    if with_synonym:
        concept = with_synonym.group(1)
        synonym = with_synonym.group(2)
        frequency = with_synonym.group(3)
    else:
        no_syn = re.search('([^:]*)(,F )([\d]*)', line)
        concept = no_syn.group(1)
        synonym = ''
        frequency = no_syn.group(3)

    print concept, synonym, frequency
