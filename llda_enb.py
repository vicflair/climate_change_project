#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Labeled LDA using the earth negotiations bulletin (ENB) as dataset
# Processing and pipeline code written by Victor Ma, but uses LLDA code
# written by Nakatani Shuyo
#
# @author: Victor Ma
# Date: 14 Apr 2014

import csv
import numpy
import pickle
import re
import string
import subprocess
import sys
import warnings
from llda import LLDA
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from optparse import OptionParser


def process_kpex_concepts(kpex_concepts_file, kpex_variants_file, taxonomy):
    synonyms = {}
    frequencies = {}
    with open(kpex_concepts_file, 'r') as f:
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
            frequency = no_syn.group(3)
        frequencies[concept] = int(frequency)

    synonyms = add_kpex_variants(kpex_variants_file, synonyms, taxonomy)
    return frequencies, synonyms


def add_kpex_variants(fname, synonyms, taxonomy):
    """ Updates synonyms rules (dict type) which transforms keys, i.e.,
    synonyms, to their main forms using the KPEX term variants.

    Inputs:
    fname - filepath of the text file containing the verbose output of KPEX's
            identification of variants
    synonyms: dict of synonyms as keys and their transformation as dict values

    Outputs:
    synonyms - input synonyms is MUTATED so this output is just a formality
    """
    with open(fname, 'r') as f:
        data = f.readlines()

    # Matches variant terms in each line
    regexp = re.compile('(?<= )[\w\- ]+(?= [\d]+\.0)')

    # Add to variants (first and second forms) to synonyms dict of
    # transformations.
    #
    # If the 1st form is already a left side in synonyms list, and its
    # replacement is not the same as in the synonyms list, then add the rule
    # "2nd form --> synonyms[1st form]" so that both the 1st and 2nd forms
    # share the same replacement.
    #
    # If only the 2nd form is the left side in the synonyms list, then add
    # the rule "1st form --> synonyms[2nd form]" so that the 1st and 2nd
    # forms share the same replacement.
    #
    # If neither form is a key, i.e. left side in the list of synonym
    # replacement rules, then add this new rule "1st form --> 2nd form" to
    # the synonyms list.
    for row in data:
        variants = re.findall(regexp, row)
        variants[0] = variants[0].lower().replace(' ', '_')
        variants[1] = variants[1].lower().replace(' ', '_')
        if variants[0] in synonyms:
            if variants[1] == synonyms[variants[0]]:
                pass
            else:
                synonyms[variants[1]] = synonyms[variants[0]]
                # print 'Case 1: ' + variants[1] + ' _> syn[' + \
                #     variants[0] + '] = ' + synonyms[variants[0]]
        elif variants[1] in synonyms:
            synonyms[variants[0]] = synonyms[variants[1]]
            # print 'Case 2: ' + variants[0] + ' _> syn[' + variants[1] + \
            #     '] = ' + synonyms[variants[1]]
        else:
            synonyms[variants[0]] = variants[1]
            # print 'Case 3: ' + variants[0] + ' _> ' + variants[1]


    # FIXME: Probably need to not only add these variants to the synonyms list,
    # but also to the frequencies list. But maybe not, since I use FreqDist()
    # anyway.
    return synonyms


def replace_synonyms(synonyms, text):
    """ Assume text is already lowercase WITHOUT n-grams, i.e. multi-word
    terms with underscores, where as synonyms is a dict WITH n-grams.
    """
    for syn in synonyms:
        text = text.replace(' ' + syn.replace('_', ' ') + ' ',
                            ' ' + synonyms[syn].replace('_', ' ') + ' ')
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
    report _ text string of entire ENB report

    Output:
    report_ngram _ input with underscores to denote concept n-grams
    """

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


def prepare_training_set(enb_file, synonyms, frequencies, taxonomy):
    """ Sanitizes and formats ENB corpus as a list of documents. Each document
    is represented as a list of word tokens, all lowercase. Stop words are
    removed. Concept n-grams are denoted by chaining using underscores, e.g.,
    "sea level rise" => "sea_level_rise"

    Input arguments:
    fname _ filepath to ENB corpus

    Output arguments:
    corpus _ list of lists contain word tokens for each report
    """
    stop = stopwords.words('english')
    corpus = []
    # Get ENB reports
    with open(enb_file) as f:
        data = csv.reader(f, delimiter='\t')
        reports = [row[7].lower() for row in data]
    for report in reports:
        # Replace synonyms with main term (according to KPEX) in both corpus
        # and taxonomy (i.e. make one "standard" form)
        report = replace_synonyms(synonyms, report)
        assert type(taxonomy) is dict or list
        if type(taxonomy) is dict:
            for concept in taxonomy:
                for index, term in enumerate(concept):
                    if term in synonyms:
                        taxonomy[concept][index] = synonyms[term]
        elif type(taxonomy) is list:
            for index, concept in enumerate(taxonomy):
                if concept in synonyms:
                    taxonomy[index] = synonyms[concept]
        # Chain unigrams into n-grams
        report = make_taxonomic_ngrams(taxonomy, report)
        report = make_kpex_ngrams(frequencies, report, 50)
        doc = []
        # Remove stop words, non-words, non-ASCII characters
        report = remove_non_ascii(report)
        for sent in sent_tokenize(report):
            doc += [word for word in word_tokenize(sent) if word not in stop]
        words = [x for x in doc if x[0] in string.ascii_letters]
        corpus.append(words)
    with open('enb_corpus', 'w') as f:
        pickle.dump(corpus, f)
        print 'Wrote prepared corpus to \'enb_corpus\''
    return corpus


def remove_non_ascii(text):
    ascii_chars = map(lambda s: s if s in string.printable else ' ', text)
    ascii_only_string = ''.join(ascii_chars)
    ascii_only_text = re.sub(' +', ' ', ascii_only_string)
    return ascii_only_text


def prepare_taxonomy(fname, cluster=False):
    """ Prepare a taxonomy file contained a list of concepts (strings) which
    have been pickled.

    Input:
    fname _ filepath to taxonomy file, a text file with one taxonomic concept
                per line

    Output:
    taxonomy _ list of taxonomic concepts as strings, normalized and sanitized
    """
    with open(fname, 'r', ) as f:
        data = csv.reader(f, delimiter='\t')
        rows = [row for row in data]
    terms = [row[1].lower().replace(' ', '_') for row in rows]
    concepts = [row[2].lower().replace(' ', '_') for row in rows]
    terms.remove('label')
    concepts.remove('clustername')

    if cluster:
        taxonomy = {}
        for term, concept in zip(terms, concepts):
            if concept in taxonomy:
                taxonomy[concept].append(term)
            else:
                taxonomy[concept] = [term]
    else:
        taxonomy = terms
    with open('taxonomy', 'w') as f:
        pickle.dump(taxonomy, f)
        print 'Wrote prepared taxonomy to \'taxonomy\''
    return taxonomy


def create_labelset(taxonomy, frequencies, corpus, threshold=30,
                    mode='count'):
    """ Create list of frequently occurring concepts in the corpus to be used
    as labels.

    Input arguments:
    taxonomy _ taxonomy data as produced by prepare_taxonomy()
    corpus _ corpus data s produced by prepare_corpus()
    threshold _ cut-off for inclusion in label set for L-LDA
    mode _ options:
            'count' (default) _ threshold by # of occurrences
            'freq' _ threshold by proportional frequency, 0.0 to 1.0

    Output arguments:
    labelset _ list of strings, representing concept-labels for L-LDA
    """
    # Filter by count or frequency
    labelset = []
    fd = FreqDist(reduce(list.__add__, corpus))
    print ''
    print 'Concept : # occurrences'
    print '___________-'
    if type(taxonomy) is dict:
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
    elif type(taxonomy) is list:
        if mode is 'count':
            for concept in taxonomy:
                occurrences = 0
                # Prefer to use KPEX frequency count rather than NLTK
                if concept in frequencies:
                    occurrences = frequencies[concept]
                else:
                    occurrences = fd[concept]
                print concept, ': #', occurrences
                if occurrences > threshold:
                    labelset.append(concept)
        elif mode is ' freq':
            for concept in taxonomy:
                # FIXME: doesn't  do frequency yet for KPEX
                concept_freq = fd.freq(concept)
                if concept_freq > threshold:
                    labelset.append(concept)
        else:
            warnings.warn('Mode option is invalid.')
    else:
        warnings.warn('Taxonomy is not a valid data type.')
    with open('labelset', 'w') as f:
        pickle.dump(labelset, f)
        print 'Wrote created labelset to \'labelset\''

    # Remove any duplicates, just in case
    labelset = list(set(labelset))
    # Remove 'common' if present, as this is a reserved label for the Shuyo
    # implementation of L-LDA
    if 'common' in labelset:
            labelset.remove('common')
    return labelset


def assign_labels(labelset, taxonomy, corpus):
    """ Given a label set and a corpus of documents, assign to each document
    the labels according to the presence of the concept-label within the
    document text.

    Inputs:
    labelset _ set of labels as produced by create_labelset()
    corpus _ corpus data as produced by prepare_corpus()

    Outputs:
    labels _ list of lists, each sub-list contains the labels (or empty list)
                for each document in the corpus
    """
    labels = []
    for document in corpus:
        document_labels = []
        # Taxonomy could be based on concept clusters (then a dict) or
        # all concepts as potential labels themselves (then a tuple)
        if type(taxonomy) is list:
            document_labels = [label for label in labelset
                               if label in document]
            labels.append(document_labels)
        elif type(taxonomy) is dict:
            presence = [term for term in taxonomy[label] if term in document]
            if presence:
                labels.append(document_labels)
    with open('labels', 'w') as f:
        pickle.dump(labels, f)
        print 'Wrote assigned labels to \'labels\''
    return labels


def write_training_set(corpus, labels, fname, semisupervised=False):
    with open(fname, 'w') as f:
        for i, label in enumerate(labels):
            if label:
                text = '\"' + ' '.join(corpus[i]) + '\"'
                text = text.replace('_', '')
                labels_str = ' '.join(label)
                line = ','.join([str(i), labels_str, text]) + '\n'
                f.write(line)
            elif semisupervised:
                text = '\"' + ' '.join(corpus[i]) + '\"'
                text = text.replace('_', '')
                labels_str = ' '.join(labelset)
                line = ','.join([str(i), labels_str, text]) + '\n'
                f.write(line)
            else:
                pass
    print 'wrote llda training set as ' + fname + '\n'


def llda_learn(tmt, script, training_set, output_folder):
    # Clear existing intermediate files
    remove_intermediate_files = ['rm', '-r', training_set + '.*']
    print 'Remove existing intermediate files with command: '
    print remove_intermediate_files
    subprocess.call(remove_intermediate_files)
    # Delete existing output folder, if applicable
    remove_folder = ['rm', '-r', output_folder]
    print 'Remove previously existing output folder with command: '
    print remove_folder
    subprocess.call(remove_folder)
    # Run (L-)LDA script
    command = ['java', '-jar', tmt, script, training_set, output_folder]
    print 'Run (L-)LDA with command: '
    print command
    subprocess.call(command)


def iterative_llda():
    pass


def main(args):
    enb_file = '../enb/ENB_Reports.csv'
    taxonomy_file = '../enb/ENB_Issue_Dictionaries.csv'
    kpex_concepts_file = 'enb_corpus_kpex.kpex_n9999.txt'
    kpex_variants_file = 'KPEX_ENB_term_variants.txt'
    training_file = 'llda_training_set'
    tmt_file = 'tmt-0.4.0.jar'
    llda_script = '6-llda-learn.scala'
    taxonomy = prepare_taxonomy(taxonomy_file, cluster=False)
    frequencies, synonyms = process_kpex_concepts(kpex_concepts_file,
                                                  kpex_variants_file, taxonomy)
    corpus = prepare_training_set(enb_file, synonyms, frequencies, taxonomy)
    labelset = create_labelset(taxonomy, frequencies, corpus)
    labels = assign_labels(labelset, taxonomy, corpus)
    write_training_set(corpus, labels, training_file, semisupervised=False)
    llda_learn(tmt_file, llda_script, training_file, args[1])


if __name__ == '__main__':
    main(sys.argv)
