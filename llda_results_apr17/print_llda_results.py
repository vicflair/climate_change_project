import pickle
import numpy

# Load vocabulary, phi (per-label word distribution), and set of labels
with open('vocas', 'r') as f:
    vocas = pickle.load(f)
with open('phi', 'r') as f:
    phi = pickle.load(f)
with open('labelset', 'r') as f:
    labelset = pickle.load(f)

# Report top N words for in each per-label word distribution
N = 50  # Number of words to report for each label
with open('top_'+str(N)+'_results.csv', 'w') as f:
    for k, label in enumerate(labelset):
        f.write(label + ', ')
    f.write('\n')
    # sort is word indices for each phi, sorted
    sort = [numpy.argsort(-phi0) for phi0 in phi]
    for i in range(N):
        for j, label in enumerate(labelset):
            # ith is ith highest word for phi[j]
            ith = sort[j+1][i]  # j+1 because 0 is the 'common' label
            # f.write(str(phi[j+1][ith]))
            f.write(vocas[ith])
            f.write(', ')
        f.write('\n')

    # Report words which were already in the taxonomy
    with open('taxonomy', 'r') as f:
        taxonomy = pickle.load(f)
    for j, label in enumerate(labelset):
        for i in range(N):
            ith = sort[j+1][i]
            for concept in taxonomy:
                if vocas[ith] in taxonomy[concept]:
                    print '*',label, '||', vocas[ith]
