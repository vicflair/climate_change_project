#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Labeled LDA using the earth negotiations bulletin (ENB) as dataset
# Based on code written by Nakatani Shuyo
#
# @author: Victor Ma
# Date: 14 Apr 2014

import csv
import sys
import string
import warnings
import numpy
import pickle
import re
import time
from llda import LLDA
from optparse import OptionParser
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

warnings.simplefilter('always')


def process_kpex_output(fname):
    synonyms = {}
    frequencies = {}
    with open(fname, 'r') as f:
        data = f.readlines()
    for line in [x.lower() for x in data]:
        regexp = re.compile('([^:]*)::syn::([\w]*),f ([\d]*)')
        with_synonym = re.search(regexp, line)
        if with_synonym:
            concept = with_synonym.group(1).replace(' ', '_')
            synonym = with_synonym.group(2).replace(' ', '_')
            frequency = with_synonym.group(3)
            synonyms[synonym] = concept
        else:
            regexp = re.compile('([^:]*)(,f )([\d]*)')
            no_syn = re.search(regexp, line)
            concept = no_syn.group(1).replace(' ', '_')
            synonym = ''
            frequency = no_syn.group(3)
        frequencies[concept] = int(frequency)
    return frequencies, synonyms


def replace_synonyms(synonyms, text):
    """ assume text is already lowercase, synonyms is a dict"""
    for syn in synonyms:
        text = text.replace(' '+syn+' ', ' '+synonyms[syn]+' ')
    return text


def make_kpex_ngrams(frequencies, text, threshold=50):
    """ Turn unigram phrases into n-grams if co-occurrence count is greater
    than threshold. text is assumed to be already lowercase. """
    for concept in frequencies:
        if frequencies[concept] > threshold:
            unigrams = concept.replace('_', ' ')
            text = text.replace(unigrams, concept)
    return text


def make_taxonomic_ngrams(taxonomy, report):
    """Chains occurrences of unigrams found as phrases in the taxonomy into
    concept n-grams, e.g., "climate change" => "climate_change".

    Inputs:
    report -- text string of entire ENB report

    Output:
    report_ngram -- input with underscores to denote concept n-grams
    """

    # Remove underscores from taxonomic concepts and find occurrences.
    # taxonomy = [re.sub('_', ' ', word) for word in taxonomy]
    # concept_matches = [concept for concept in taxonomy if
    #                    report.find(concept) is not -1]

    # Replace word chains with n-grams as denoted with an underscore, but check
    # first whether taxonomy is based on clusters or all concepts separately.
    report_ngram = report
    if type(taxonomy) is dict:
        for cluster in taxonomy:
            ngram_matches = [concept for concept in taxonomy[cluster] if
                concept.replace('_', ' ') in report]
            for ngram in ngram_matches:
                unigrams = ngram.replace('_', ' ')
                report_ngram = report_ngram.replace(unigrams, ngram)
    elif type(taxonomy) is list:
        ngram_matches = [concept for concept in taxonomy
                         if concept.replace('_', ' ') in report_ngram]
        for ngram in ngram_matches:
            unigrams = ngram.replace('_', ' ')
            report_ngram = report_ngram.replace(unigrams, ngram)
    else:
        warnings.warn('Invalid taxonomy data type.')
    return report_ngram

def prepare_enb_corpus2(fname, synonyms, frequencies, taxonomy):
    stop = stopwords.words('english')
    corpus = []
    # Get ENB reports
    with open(fname) as f:
        data = csv.reader(f, delimiter='\t')
        reports = [row[7].lower() for row in data]
    for report in reports:
        # Replace synonyms with main term (according to KPEX)
        report = replace_synonyms(synonyms, report)
        # Chain unigrams into n-grams
        report = make_taxonomic_ngrams(taxonomy, report)
        report = make_kpex_ngrams(frequencies, report, 50)
        doc = []
        for sent in sent_tokenize(report):
            doc += [word for word in word_tokenize(sent) if word not in stop]
        corpus.append([x for x in doc if x[0] in string.ascii_letters])
    return corpus

def prepare_enb_corpus(fname):
    """ Sanitizes and formats ENB corpus as a list of documents. Each document
    is represented as a list of word tokens, all lowercase. Stop words are
    removed. Concept n-grams are denoted by chaining using underscores, e.g.,
    "sea level rise" => "sea_level_rise"

    Input arguments:
    fname -- filepath to ENB corpus

    Output arguments:
    corpus -- list of lists contain word tokens for each report
    """
    stop = stopwords.words('english')
    corpus = []
    # Clean and format corpus
    with open(fname) as f:
        data = csv.reader(f, delimiter='\t')
        reports = [row[7] for row in data]  # Index 7 is report text
        for report in reports:
            report = report.lower()
            report = make_taxonomic_ngrams(report)
            doc = []
            for sent in sent_tokenize(report):
                doc += [word for word in word_tokenize(sent)
                        if word not in stop]
            corpus.append([x for x in doc if x[0] in string.ascii_letters])

    # Serialize prepared corpus and save to file
    with open('enb_corpus', 'w') as f:
        pickle.dump(corpus, f)
        print 'Wrote prepared corpus to \'enb_corpus\''
    return corpus


def prepare_taxonomy(fname):
    """ Prepare a taxonomy file contained a list of concepts (strings) which
    have been pickled.

    Input:
    fname -- filepath to taxonomy file, a text file with one taxonomic concept
                per line

    Output:
    taxonomy -- list of taxonomic concepts as strings, normalized and sanitized
    """

    # Normalize and sanitize taxonomy.
    with open(fname) as f:
        taxonomy = f.readlines()
    taxonomy = [word.rstrip() for word in taxonomy]
    taxonomy = [word.lower() for word in taxonomy]
    taxonomy = [re.sub(' ', '_', word) for word in taxonomy]

    # Remove duplicates (previously caused problems with L-LDA code).
    taxonomy = list(set(taxonomy))

    # Serialize taxonomy and save to file.
    with open('enb_taxonomy', 'w') as f:
        pickle.dump(taxonomy, f)
        print 'Wrote prepared taxonomy to \'enb_taxonomy\''
    return taxonomy


def prepare_taxonomy2(fname):
    with open(fname, 'r',) as f:
        data = csv.reader(f, delimiter='\t')
        rows  = [row for row in data]
    terms = [row[1].lower().replace(' ', '_') for row in rows]
    concepts = [row[2].lower().replace(' ', '_') for row in rows]
    terms.remove('label')
    concepts.remove('clustername')

    taxonomy = {}
    for term, concept in zip(terms, concepts):
        if concept in taxonomy:
            taxonomy[concept].append(term)
        else:
            taxonomy[concept] = [term]
    with open('taxonomy', 'w') as f:
        pickle.dump(taxonomy, f)
        print 'Wrote prepared taxonomy to \'taxonomy\''
    return taxonomy


def create_labelset2(taxonomy, frequencies, corpus, threshold=60, mode='count'):
    # Filter by count or frequency
    labelset = []
    fd = FreqDist(reduce(list.__add__, corpus))
    print ''
    print 'Concept : # occurrences'
    print '-----------------------'
    for concept in taxonomy:
        if mode is 'count':
            terms_count = 0
            for term in taxonomy[concept]:
                # Prefer to use KPEX frequency count rather than NLTK
                if term in frequencies:
                    terms_count += frequencies[term]
                else:
                    terms_count += fd[term]
            print concept, ': #', terms_count
            if terms_count > threshold:
                labelset.append(concept)
        elif mode is 'freq':
            # FIXME: Doesn't do frequency yet for KPEX
            terms_freq = sum([fd.freq(term) for term in concept])
            if terms_freq > threshold:
                labelset.append(concept)
    with open('labelset', 'w') as f:
        pickle.dump(labelset, f)
        print 'Wrote created labelset to \'labelset\''
    return labelset


def assign_labels2(labelset, taxonomy, corpus):
    labels = []
    for document in corpus:
        document_labels = []
        for label in labelset:
            # Taxonomy could be based on concept clusters (then a dict) or
            # all concepts as potential labels themselves (then a tuple)
            if type(taxonomy) is dict:
                presence = [term for term in taxonomy[label]
                            if term in document]
                if presence:
                    document_labels.append(label)
            elif type(taxonomy) is tuple:
                document_labels = [label for label in labelset
                                   if label in document]
        labels.append(document_labels)
    with open('labels', 'w') as f:
        pickle.dump(labels, f)
        print 'Wrote assigned labels to \'labels\''
    return labels


def create_labelset(taxonomy, corpus, threshold, mode='count'):
    """ Create list of frequently occurring concepts in the corpus to be used
    as labels.

    Input arguments:
    taxonomy -- taxonomy data as produced by prepare_taxonomy()
    corpus -- corpus data s produced by prepare_corpus()
    threshold -- cut-off for inclusion in label set for L-LDA
    mode -- options:
            'count' (default) -- threshold by # of occurrences
            'freq' -- threshold by proportional frequency, 0.0 to 1.0

    Output arguments:
    labelset -- list of strings, representing concept-labels for L-LDA
    """

    # Filter by count or frequency
    labelset = []
    fd = FreqDist(reduce(list.__add__, corpus))
    for concept in taxonomy:
        if (mode is 'count') and (fd[concept] > threshold):
            labelset.append(concept)
        elif (mode is 'freq') and (fd.freq(concept) > threshold):
            labelset.append(concept)

    # Serialize and save label set to file
    with open('labelset', 'w') as f:
        pickle.dump(labelset, f)
        print 'Wrote created labelset to \'labelset\''
    return labelset


def assign_labels(labelset, corpus):
    """ Given a label set and a corpus of documents, assign to each document
    the labels according to the presence of the concept-label within the
    document text.

    Inputs:
    labelset -- set of labels as produced by create_labelset()
    corpus -- corpus data as produced by prepare_corpus()

    Outputs:
    labels -- list of lists, each sub-list contains the labels (or empty list)
                for each document in the corpus
    """
    labels = []
    for document in corpus:
        labels.append([label for label in labelset if label in document])
    return labels


def run_llda(labelset, corpus, labels):
    """" Performs L-LDA given a label set, a corpus, and the assigned labels
    from that label set for the documents in that corpus. Currently, L-LDA
    parameters and settings are only modifiable within the function code.

    Inputs:
    labelset -- produced by create_labelset()
    corpus -- produced by prepare_corpus()
    labels -- produced by assign_labels()

    Output:
    None
    """

    # Set up LLDA parameters and run
    parser = OptionParser()
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=20)
    parser.add_option("-s", dest="seed", type="int", help="random seed", default=None)
    parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
    (options, args) = parser.parse_args()

    llda = LLDA(options.K, options.alpha, options.beta)
    llda.set_corpus(labelset, corpus, labels)

    # Show progress
    print "M=%d, V=%d, L=%d, K=%d" % (len(corpus), len(llda.vocas), len(labelset), options.K)

    for i in range(options.iteration):
        sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
        llda.inference()

        # Incrementally save results
        phi = llda.phi()
        with open('phi', 'w') as f:
            pickle.dump(phi, f)
        # with open('vocas', 'w') as f:
            # pickle.dump(llda.vocas, f)
        with open('top_20_words.txt', 'w') as f:
            for k, label in enumerate(labelset):
                f.write('\n\n-- label ' + str(k) + ' : ' + label)
                for w in numpy.argsort(-phi[k])[:20]:
                    f.write('\n' + llda.vocas[w] + ': ' + str(phi[k,w]))
    print "perplexity : %.4f" % llda.perplexity()

    # Final reporting and saving of L-LDA results
    phi = llda.phi()
    with open('phi', 'w') as f:
        pickle.dump(phi, f)
        print 'Saved LLDA results (phi) to \'phi\''
    with open('vocas', 'w') as f:
        pickle.dump(llda.vocas, f)
        print 'Saved LLDA results (vocas) to \'vocas\''
    with open('top_20_words.txt', 'w') as f:
        for k, label in enumerate(labelset):
            print "\n-- label %d : %s" % (k, label)
            f.write('\n\n-- label ' + str(k) + ' : ' + label)
            for w in numpy.argsort(-phi[k])[:20]:
                print "%s: %.4f" % (llda.vocas[w], phi[k,w])
                f.write('\n' + llda.vocas[w] + ': ' + str(phi[k,w]))


def llda_v1():
    # Create data if not already done
    corpus = prepare_enb_corpus('../ENB/ENB_Reports.csv')
    taxonomy = prepare_taxonomy('../ENB/enb_expression_clusters.txt')
    labelset = create_labelset(taxonomy, corpus, 50)
    labels = assign_labels(labelset, corpus)

    # Run LLDA with default options
    run_llda(labelset, corpus, labels)

def llda_v2():
    # Try LLDA with clusters instead of all concept terms, and time
    start = time.time()
    taxonomy = prepare_taxonomy2('../ENB/ENB_Issue_Dictionaries.csv')
    corpus = prepare_enb_corpus('../ENB/ENB_Reports.csv')
    labelset = create_labelset2(taxonomy, corpus, 100)
    labels = assign_labels2(labelset, taxonomy, corpus)
    run_llda(labelset, corpus, labels)
    end = time.time()
    with open('llda_time.txt', 'w') as f:
        f.write(str(end-start) + ' secs')


def llda_v3():
    # LLDA with clusters and KPEX n-grams
    enb_corpus_file = '../enb/ENB_Reports.csv'
    taxonomy_file = '../enb/ENB_Issue_Dictionaries.csv'
    kpex_output_file = 'enb_corpus_kpex.kpex_n9999.txt'
    frequencies, synonyms = process_kpex_output(kpex_output_file)
    taxonomy = prepare_taxonomy2(taxonomy_file)
    corpus = prepare_enb_corpus2(enb_corpus_file, synonyms, frequencies,
                                 taxonomy)
    labelset = create_labelset2(taxonomy, frequencies, corpus, 60)
    labels = assign_labels2(labelset, taxonomy, corpus)


def main():
    pass


if __name__ == '__main__':
    main()
