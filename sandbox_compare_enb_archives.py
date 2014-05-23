import csv
import difflib
import pickle

with open('../enb/ENB_Reports.csv', 'r') as f:
    data = csv.reader(f, delimiter='\t')
    enb_reports = [row[7] for row in data]

with open('enb_archives_texts', 'r') as f:
    enb_archives = pickle.load(f)

matching_seqs = []
for i, rep in enumerate(enb_reports):
    max_match_size = -1
    best_match = []
    for j, arc in enumerate(enb_archives):
        sq = difflib.SequenceMatcher(isjunk=None, a=rep, b=arc)
        alo = 0
        ahi = len(rep)
        blo = 0
        bhi = len(arc)
        match = sq.find_longest_match(alo, ahi, blo, bhi)
        if match.size > max_match_size:
            max_match_size = match.size
            best_match = [(j, match)]
        elif match.size == max_match_size:
            best_match.append((j, match))
    print '%d: Best match size = %d' % (i, max_match_size)
    print map(lambda t: t[0], best_match)
    print ''
    matching_seqs.append((i, best_match))
with open('matching_seqs', 'w') as f:
    pickle.dump(matching_seqs, f)
