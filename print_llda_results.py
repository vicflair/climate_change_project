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
N = 20  # Number of words to report for each label
for k, label in enumerate(labelset):
    print '\n -- label %d : %s' % (k, label)
    for w in numpy.argsort(-phi[k][:N]):
        print "%s: %.4f" % (vocas[w], phi[k, w])
