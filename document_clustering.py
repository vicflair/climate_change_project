# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause
#
# Modified for research use by Victor Ma <victor.ma@epfl.ch>

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import csv
import string
import pickle
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   "to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
from llda_enb import *
# enb_file = '../enb/ENB_Reports.csv'
# taxonomy_file = '../enb/ENB_Issue_Dictionaries.csv'
# kpex_concepts_file = 'enb_corpus_kpex.kpex_n9999.txt'
# kpex_variants_file = 'KPEX_ENB_term_variants.txt'
# taxonomy = prepare_taxonomy(taxonomy_file, cluster=False)
# frequencies, synonyms = process_kpex_concepts(kpex_concepts_file,
#                                               kpex_variants_file, taxonomy)
# corpus = prepare_training_set(enb_file, synonyms, frequencies, taxonomy)
with open('enb_corpus', 'r') as f:
    corpus = pickle.load(f)

reports = map(lambda x: ' '.join(x), corpus)

# print('Loading '  + enb_file)
# with open(enb_file) as f:
#     data = csv.reader(f, delimiter='\t')
#     reports = [row[7].lower() for row in data]
labels = [0]*len(reports)
true_k = 30

print("%d documents" % len(reports))
print("%d categories" % true_k)
print()

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = Pipeline((
            ('hasher', hasher),
            ('tf_idf', TfidfTransformer())
        ))
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 stop_words='english', use_idf=opts.use_idf)

X = vectorizer.fit_transform(reports)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    lsa = TruncatedSVD(opts.n_components)
    X = lsa.fit_transform(X)
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    X = Normalizer(copy=False).fit_transform(X)

    print("done in %fs" % (time() - t0))
    print()


###############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels, sample_size=1000))
print()

###############################################################################
####### Run L-LDA  #######

labels = map(lambda x: ['cluster_'+str(x)], list(km.labels_))
with open('taxonomy', 'r') as f:
    taxonomy = pickle.load(f)
tmt_file = 'tmt-0.4.0.jar'
llda_script = '6-llda-learn.scala'
training_file = 'labeled_by_kmeans'
output_folder = 'topics_kmeans'

write_training_set(corpus, labels, training_file, semisupervised=False)
llda_learn(tmt_file, llda_script, training_file, output_folder)

# Process results
command = ['cp', output_folder+'/01500/topic-term-distributions.csv.gz',
           output_folder+'/01500/results.csv.gz']
subprocess.call(command)
command = ['gunzip', output_folder+'/01500/results.csv.gz']
subprocess.call(command)
with open(output_folder+'/01500/term-index.txt', 'r') as f:
    terms = f.readlines()
    terms = map(str.rstrip, terms)
with open(output_folder+'/01500/label-index.txt', 'r') as f:
    labels = f.readlines()
    labels = map(str.rstrip, labels)
topics = dict()
with open(output_folder+'/01500/results.csv', 'r') as f:
    data = csv.reader(f, delimiter=',')
    data = [row for row in data]
    for i, label in enumerate(labels):
        topics[label] = dict()
        for j, term in enumerate(terms):
            topics[label][term] = float(data[i][j])

# View top N concepts
N = 15
for label in labels:
    print('')
    line = '-'*17+label+'-'*17
    print(line)
    ordered = [(term, topics[label][term]) for term in terms
               if term in taxonomy]
    ordered = sorted(ordered, key=lambda tup: tup[1], reverse=True)
    # new_ordered = ordered
    # for item in ordered:
    #     full_term = [oterm for oterm in original_taxonomy
    #                  if oterm.replace('_', '') == item[0]]
    #     new_ordered.append((full_term[0], item[1]))
    for i in range(N):
        #print new_ordered[i][0], str(round(new_ordered[i][1],1)) + '%'
        line = '%40s %5s' % (ordered[i][0], str(round(ordered[i][1], 1)) + '%')
        print(line)

# Write to file
N = 15
with open('results_cluster_kmeans.txt', 'w') as f:
    for label in labels:
        line = '-'*17+label+'-'*17 + '\n'
        f.write(line)
        ordered = [(term, topics[label][term]) for term in terms
                   if term in taxonomy]
        ordered = sorted(ordered, key=lambda tup: tup[1], reverse=True)
        # new_ordered = ordered
        # for item in ordered:
        #     full_term = [oterm for oterm in original_taxonomy
        #                  if oterm.replace('_', '') == item[0]]
        #     new_ordered.append((full_term[0], item[1]))
        for i in range(N):
            #print new_ordered[i][0], str(round(new_ordered[i][1],1)) + '%'
            line = '%40s %5s' % (ordered[i][0], str(round(ordered[i][1], 1)) + '%')
            f.write(line+'\n')
        f.write('\n')

###############################################################################
# # VMa Post-analysis
# stop = stopwords.words('english')
# count = [0]*true_k
# for i in km.labels_:
#     count[i] += 1
#
# # Group reports by topics
# topics = []
# for i in range(true_k):
#     topics.append([report for j, report in enumerate(reports)
#                   if km.labels_[j] == i])
#
# # Get top N terms for each topic
# tokens = []
# for topic in topics:
#     all_words = []
#     for report in topic:
#         # report = remove_non_ascii(report)
#         doc = []
#         for sent in sent_tokenize(report):
#             doc += [word for word in word_tokenize(sent) if word not in stop]
#         words = [x for x in doc if x[0] in string.ascii_letters]
#         all_words += words
#     tokens.append(all_words)
#
# for token in tokens:
#     fd = FreqDist(token)
#     print('')
#     for i in range(0, 30):
#         print(fd.keys()[i], fd.values()[i])
#
# # For each topic (set of tokens), get the frequency count of all concepts in
# # taxonomy. Get top N concepts for each topic. Rank and display
# for token in tokens:
#     concepts = [term for term in token if term in taxonomy]
#     fd = FreqDist(concepts)
#     concept_counts = []
#     for key, value in zip(fd.keys(), fd.values()):
#         freq = round(100.0*value/fd.N(), 1)
#         info = (key, value, freq)
#         concept_counts.append(info)
#     print('')
#     for i in range(0, 15):
#         print(concept_counts[i][0], str(concept_counts[i][2])+'%')
