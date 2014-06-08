#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Labeled LDA using the earth negotiations bulletin (ENB) as dataset
# Processing and pipeline code written by Victor Ma, but uses LLDA code
# from the Stanford TMT.
#
# @author: Victor Ma
# Date: 14 Apr 2014

import csv
import numpy
import ntpath
import os
import pickle
import re
import string
import subprocess
import sys
import time
import warnings
from functools import wraps
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize


lemmatizer = WordNetLemmatizer()


def timeit(func):
    """ Decorator function used to time execution.
    """
    @wraps(func)
    def timed_function(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print '%s execution time: %f secs' % (func.__name__, end-start)
        return output
    return timed_function


def memorize(func):
    """ Decorator function for caching outputs of a function to speed up
     retrieval time. If an argument has been previously used, the corresponding
     output will be looked up instead of calculated.
    """
    # TODO: Figure out how this cache is maintained. Why doesn't it disappear?
    cache = {}
    @wraps(func)
    def cached_function(*args, **kwargs):
        if args not in cache:
            cache[args] = func(*args, **kwargs)
        return cache[args]
    return cached_function


def replace_enb_unicode(text):
    """ Replace non-ASCII unicode characters in ENB corpus using manually
    determined rules. """
    replacement = [
        # for Ida Karnstrom
        ['K\xc3\x83\xc2\xa4rnstr\xc3\x83\xc2\xb6m', 'Karnstrom'],
        # for Raul Estrada-Oyuela
        ['Ra\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdl', 'Raul'],
        # for Klaus Topfer
        ['T\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdpfer', 'Topfer'],
        # for Bo Kjellen
        ['Kjell\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdn', 'Kjellen'],
        # for COTE D'IVOIRE
        ['C\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdTE', 'COTE'],
        # for Mans Lonroth
        ['L\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdnroth', 'Lonroth'],
        # for Carlos Gomez
        ['G\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdmez', 'Gomez'],
        # for Thomas Bucher
        ['B<F"Times New Roman">\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbd<F255>cher', 'Bucher'],
        # for Alvaro Umana
        ['Uma\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbda', 'Umana'],
        # for Jorge Berguno
        ['Bergu\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdo', 'Berguno'],
        # for Antonio La Vina
        ['Vi\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbda', 'Vina'],
        # for Abdelaziz Saadi
        ['Sa\xc3\xa2di', 'Saadi'],
        ['\xc3\xa2\xc2\x80\xc2\x9c', '\"'],
        ['\xc3\xa2\xc2\x80\xc2\x9d', '\"'],
        ['\xc3\xa2\xc2\x80\xc2\x99', '\''],
        ['\xc3\xa2\xc2\x80\xc2\x93', '-'],
        ['\xc3\xaf\xc2\xbf\xc2\xbd', '\''],
        ['\xc3\x82\c2\xb0', ''],
        ['\xc3\x8b\xc2\x9a', ''],
        ['\xc3\x82\xc2\xa0', ''],
        ['\xc3\x83\xc2\x85', 'A'],
        ['\xc3\x83\xc2\xa9', 'e'],
        ['\xc3\x83\xc2\xa0', 'a'],
        ['\xc3\x83\xc2\xa5', 'a'],
        ['\xe2\x82\xac', 'EUR '],  # euro symbol
        ['\xc2\xa3', 'GBP '],  # pound symbol
        ['\xe2\x80\xaf', ' '],
        ['\xe2\x80\xa6', '...'],
        ['\xc2\xb1', 'n'],
        ['\xc2\xba', ''],
        ['\xc2\x8c', 'i'],
        ['\xc2\x82', 'e'],
        ['\xc2\x80', ''],
        ['\xc2\x97', ''],
        ['\xc2\x89', 'E'],
        ['\xc3\x89', 'E'],
        ['\xc2\x93', ''],  # seems to be paired with \xc2\x93, maybe quotes?
        ['\xc2\x94', ' '],
        ['\xc2\x91', '\''],
        ['\xc2\x92', '\''],
        ['\xc2\xb4', '\''],
        ['\xc2\xa8', '\"'],
        ['\xc5\x84', 'n'],
        ['\xc3\xb1', 'n'],
        ['\xc3\xa2', ''],
        ['\xc2\x9c', ''],
        ['\xc2\x9d', ''],
        ['\xc2\xa6', ''],
        ['\xc3\x83', 'A'],
        ['\xc2\x85', ''],
        ['\xc3\x9c', 'U'],
        ['\xc3\x82 C', ' degrees C'],  # sometimes used as degree symbol
        ['\xc3\x82', ''],  # if not degree, discard
        ['\xc2\xb0', ' '],  # degree symbol, good replacement?
        ['\xc3\x94', 'O'],
        ['\xc3\xa9', 'e'],
        ['\xc3\xaa', 'e'],
        ['\xc3\xa9', 'e'],
        ['\xc4\x87', 'c'],
        ['\xc3\xa8', 'e'],
        ['\xc3\xb3', 'o'],
        ['\xc3\xb6', 'o'],
        ['\xc3\xa1', 'a'],
        ['\xc3\xad', 'i'],
        ['\xcc\x81', 'n'],
        ['\xc2\xad', ''],  # Used as space, but incorrectly
        ['\xc3\xa4', 'a'],
        ['\xc3\xba', 'u'],
        ['\xc3\xb8', 'o'],
        ['\xc3\xbc', 'u'],
        ['\xc2\x96', ''],
        ['\xc2\xb8', '']  # used only once, incorrectly as comma
    ]
    for r in replacement:
        if r[0] in text:
            text = text.replace(r[0], r[1])
    return text


@memorize
def lemmatize_word(word, pos_tag='n'):
    lemma = lemmatizer.lemmatize(word, pos_tag)
    return lemma


def process_kpex_concepts(kpex_concepts_file, kpex_variants_file=None,
                          taxonomy=None, threshold=0):
    synonyms = {}
    frequencies = {}
    with open(kpex_concepts_file, 'r') as f:
        data = f.readlines()
    for line in [x.lower().split(',') for x in data]:
        # Extract KPEX concept/n-gram
        concept_match = re.search('[^:]*', line[0])
        concept = concept_match.group().replace(' ', '_')
        # Extract KPEX score for ranking concepts/n-grams
        if len(line) is 3:
            # TODO: Currently, the score is unused.
            score_match = re.search('[\d\.]*', line[1])
            score = float(score_match.group())
            # Extract frequency/count
            frequency_match = re.search('[\d]+', line[2])
        else:
            frequency_match = re.search('[\d]+', line[1])
        frequency = int(frequency_match.group())
        if frequency >= threshold:
            frequencies[concept] = frequency
            # Extract synonym, if it exists.
            synonym_match = re.search('(?<=::syn::)(.+)', line[0])
            if synonym_match:
                synonym = synonym_match.group().replace(' ', '_')
                synonyms[synonym] = concept
    if kpex_variants_file:
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


def replace_synonyms(synonyms, frequencies, text, threshold=0):
    """ Replaces synonyms with their "main" form as determined by KPEX, if
    their frequency is above threshold. Assume text is already lowercase
    WITHOUT n-grams, i.e. multi-word terms with underscores, where as
    synonyms is a dict WITH n-grams.
    """
    valid_synonyms = [syn for syn in synonyms
                      if frequencies[synonyms[syn]] >= threshold]
    for syn in valid_synonyms:
        text = text.replace(' ' + syn.replace('_', ' ') + ' ',
                            ' ' + synonyms[syn].replace('_', ' ') + ' ')
    return text


def make_kpex_ngrams(frequencies, text, threshold=0):
    """ Turn unigram phrases into n-grams if their frequency is greater than
    threshold. text is assumed to be already lowercase. This currently
    doesn't order transformations in intelligent ways, e.g., if
    'climate_change' and 'abrupt_climate_change' are both KPEX terms, then it
    will apply the n-gram transformation that comes first in the KPEX terms
    list, without consideration for n-gram length, score, frequency, etc. """
    for concept in frequencies:
        if frequencies[concept] >= threshold:
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


def extract_scientific_articles(article_files, output_filename=None):
    scientific_articles = []
    for i, file_path in enumerate(article_files):
        with open(file_path, 'r') as f:
            text = f.read()
        scientific_articles.append(text)
        print '%d of %d files extracted' % (i+1, len(article_files))
    return scientific_articles


def extract_kpex_data(kpex_files, threshold=3, output_filename=None):
    kpex_data = []
    for i, kpex_file in enumerate(kpex_files):
        frequencies, synonyms = process_kpex_concepts(kpex_file,
                                                      threshold=threshold)
        kpex_data.append((frequencies, synonyms))
        print '%d of %d files extracted' % (i+1, len(kpex_files))
    if output_filename:
        with open(output_filename, 'w') as f:
            pickle.dump(kpex_data, f)
    return kpex_data


def get_articles_and_kpex_file_paths(folder_path):
    article_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(folder_path)
                 for name in files
                 if name.endswith(("xml_marked.txt", ".text"))]
    kpex_files = [os.path.join(root, name)
                     for root, dirs, files in os.walk(folder_path)
                     for name in files
                     if name.endswith(("kpex_n9999.txt", ".text"))]
    # There are less kpex files than articles, so filter out the articles which
    # don't have a corresponding kpex files
    article_files = [file for file in article_files
                     if file.replace('.txt', '.kpex_n9999.txt') in kpex_files]
    # Sort so that they match index-wise
    article_files = sorted(article_files)
    kpex_files = sorted(kpex_files)
    return article_files, kpex_files


def process_scientific_articles(scientific_articles, kpex_data, taxonomy,
                                threshold=0):
    """ Very similar to the process_enb_reports() except it takes kpex info
    in a different format since there is one KPEX file for every scientific
    article as compared to one KPEX file for all ENB reports.

    This function is separate because it may be later than the scientific
    articles require a different set of pre-processing steps.
    """

    stop = stopwords.words('english')
    corpus = []
    for article, kpex in zip(scientific_articles, kpex_data):
        # "Unpack" and rename KPEX data items
        frequencies = kpex[0]
        synonyms = kpex[1]
        # Remove non-ASCII words and characters:
        article = remove_non_ascii(article)  # TODO: needed for articles?
        # Replace synonyms with main term (according to KPEX) in both corpus
        # and taxonomy (i.e. make one "standard" form)
        article = replace_synonyms(synonyms, frequencies, article,
                                  threshold=threshold)
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
        # Chain unigrams into n-grams (ontology first, THEN, KPEX terms)
        article = make_taxonomic_ngrams(taxonomy, article)
        article = make_kpex_ngrams(frequencies, article,
                                   threshold=threshold)
        doc = []
        # Remove stop words, non-words. Lemmatize the rest.
        for sent in sent_tokenize(article):
            doc += [word for word in word_tokenize(sent) if word not in stop]
        words = [lemmatize_word(x) for x in doc
                 if x[0] in string.ascii_letters]
        # Filter words not in taxonomy or KPEX terms list
        filtered_words = filter_terms(words, taxonomy=taxonomy,
                                      frequencies=frequencies,
                                      threshold=threshold)
        corpus.append(filtered_words)
    # Save correspondences between underscored and concatenated version of all
    # n-grams in the corpus so we can do backwards conversion after LDA
    all_terms = set(reduce(list.__add__, corpus))
    underscores = {}
    for term in all_terms:
        no_underscores = term.replace('_', '')
        underscores[no_underscores] = term
    return corpus, underscores


def process_enb_reports(enb_file, synonyms, frequencies, taxonomy,
                        threshold=0):
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
    with open(enb_file, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        reports = [row[7].lower() for row in data]
    for report in reports:
        # Remove non-ASCII characters
        report = remove_non_ascii(report)
        # Replace synonyms with main term (according to KPEX) in both corpus
        # and taxonomy (i.e. make one "standard" form)
        report = replace_synonyms(synonyms, frequencies, report,
                                  threshold=threshold)
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
        # Chain unigrams into n-grams (ontology first, THEN, KPEX terms)
        report = make_taxonomic_ngrams(taxonomy, report)
        report = make_kpex_ngrams(frequencies, report,
                                  threshold=threshold)
        doc = []
        # Remove stop words, non-words
        for sent in sent_tokenize(report):
            doc += [word for word in word_tokenize(sent) if word not in stop]
        words = [lemmatize_word(x) for x in doc
                 if x[0] in string.ascii_letters]
        # Filter words not in taxonomy or KPEX terms list
        filtered_words = filter_terms(words, taxonomy=taxonomy,
                                      frequencies=frequencies,
                                      threshold=threshold)
        corpus.append(filtered_words)
    # Save correspondences between underscored and concatenated version of all
    # n-grams in the corpus so we can do backwards conversion after LDA
    all_terms = set(reduce(list.__add__, corpus))
    underscores = {}
    for term in all_terms:
        no_underscores = term.replace('_', '')
        underscores[no_underscores] = term
    return corpus, underscores


def filter_terms(document, taxonomy=None, frequencies=None, threshold=0):
    # Keep words only in KPEX terms or ontology, depending on whether they
    # are provided as input
    # Create list of words to keep
    keep_words = []
    if frequencies:
        keep_words = [term for term in frequencies
                      if frequencies[term] > threshold]
    if taxonomy:
        if type(taxonomy) is dict:
            for concept in taxonomy:
                keep_words += taxonomy[concept]
        elif type(taxonomy) is list:
            keep_words += taxonomy
        else:
            print 'Taxonomy is not a valid type (dict, list).'
    keep_words = list(set(keep_words))
    # Now filter words and create a filtered corpus
    if keep_words:
        filtered_document = [word for word in document if word in keep_words]
    else:
        filtered_document = document
    return filtered_document


def remove_non_ascii(text):
    # Remove unprintable characters
    ascii_chars = map(lambda s: s if s in string.printable else ' ', text)
    ascii_only_string = ''.join(ascii_chars)
    # Replace \t, \n, \r with spaces
    ascii_replaced = re.sub('[\t\n\r\x0b\x0c]', ' ', ascii_only_string)
    # Condense spaces
    ascii_only_text = re.sub(' +', ' ', ascii_replaced)
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
    csv.field_size_limit(sys.maxsize)
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
        # Ensure each cluster has no duplicates, but allow duplicates across
        # different clusters (i.e these terms have multiple concepts)
        for concept in concepts:
            taxonomy[concept] = list(set(taxonomy[concept]))
    else:
        taxonomy = terms
        # Ensure taxonomy has no duplicates
        taxonomy = list(set(taxonomy))
    return taxonomy


def create_labelset(taxonomy, frequencies, corpus, threshold=0,
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
                if terms_count >= threshold:
                    labelset.append(concept)
            elif mode is 'freq':
                # FIXME: Doesn't do frequency yet for KPEX
                terms_freq = sum([fd.freq(term) for term in concept])
                if terms_freq >= threshold:
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
                if occurrences >= threshold:
                    labelset.append(concept)
        elif mode is ' freq':
            for concept in taxonomy:
                # FIXME: doesn't  do frequency yet for KPEX
                concept_freq = fd.freq(concept)
                if concept_freq >= threshold:
                    labelset.append(concept)
        else:
            warnings.warn('Mode option is invalid.')
    else:
        warnings.warn('Taxonomy is not a valid data type.')

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
            document_labels = []
            for label in labelset:
                presence = [term for term in taxonomy[label] if term in
                            document]
                if presence:
                    document_labels.append(label)
            labels.append(document_labels)
    return labels


def write_data_set(corpus, labels, fname, semisupervised=False):
    if semisupervised:
        all_labels = ' '.join(set(reduce(list.__add__, labels)))
    with open(fname, 'w') as f:
        for i, label in enumerate(labels):
            # Write label(s) for each document
            if label:
                text = '\"' + ' '.join(corpus[i]) + '\"'
                text = text.replace('_', '')
                labels_str = ' '.join(label)
                line = ','.join([str(i), labels_str, text]) + '\n'
                f.write(line)
            elif semisupervised:
                text = '\"' + ' '.join(corpus[i]) + '\"'
                text = text.replace('_', '')
                labels_str = all_labels
                line = ','.join([str(i), labels_str, text]) + '\n'
                f.write(line)
            else:
                pass
    print 'wrote llda data set as ' + fname + '\n'


def llda_learn(tmt, script, training_set, output_folder):
    # Clear existing intermediate files
    remove_intermediate_files = ['rm', training_set + '.*']
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


def llda_infer(tmt, script, model_path, target_file, output_file):
    # Perform inference on unlabeled reports
    command = ['java', '-jar', tmt, script, model_path, target_file,
               output_file]
    print 'Run (L-)LDA-based inference with command: '
    print command
    subprocess.call(command)


def process_inference_results(model_path, inference_file, underscores):
    # Process results
    command = ['cp', model_path + '/01500/topic-term-distributions.csv.gz',
               model_path + '/01500/results.csv.gz']
    subprocess.call(command)
    command = ['gunzip', model_path + '/01500/results.csv.gz']
    subprocess.call(command)
    with open(model_path + '/01500/term-index.txt', 'r') as f:
        terms = f.readlines()
        terms = map(str.rstrip, terms)
    with open(model_path + '/01500/label-index.txt', 'r') as f:
        label_index = f.readlines()
        label_index = map(str.rstrip, label_index)
    topics = dict()
    with open(model_path + '/01500/results.csv', 'r') as f:
        data = csv.reader(f, delimiter=',')
        data = [row for row in data]
        for i, label in enumerate(label_index):
            topics[label] = dict()
            for j, term in enumerate(terms):
                topics[label][underscores[term]] = float(data[i][j])

    # Get ranked labels for each report
    with open(inference_file, 'r') as f:
        data = csv.reader(f, delimiter=',')
        inferences = [(int(row[0]), map(float, row[1:])) for row in data]
    ranked_labels = []
    for inference in inferences:
        id_num = inference[0]
        label_distribution = zip(label_index, inference[1])
        ordered_labels = sorted(label_distribution, key=lambda t: t[1],
                                     reverse=True)
        ranked_labels.append((id_num, ordered_labels))

    return topics, ranked_labels


def write_inference_results(topics, ranked_labels, output_folder):
    # TODO: SHOW TF-IDF weighted topic vectors

    # Write top N labels for each report
    N = 3
    with open(output_folder + '/TOP_3_TOPICS_PER_ENB_DOCUMENT.txt', 'w') as f:
        for id_num, ordered_labels in ranked_labels:
            f.write('Report ID# %d\n' % id_num)
            f.write('{:>25} {:>20}-\n'.format('--Topics--', '--Proportion-'))
            for i in range(N):
                label = ordered_labels[i][0]
                probability = round(100 * ordered_labels[i][1], 2)
                line = '{:>25} {:>20}%\n'.format(label, probability)
                f.write(line)
            f.write('\n')

    # Write counts for # of documents with each label in position 1:N
    N = 3
    with open(output_folder + '/TOP_3_TOPICS_COUNTS.txt', 'w') as f:
        label_set = [label for label in topics]
        for N in range(3):
            label_counts = []
            for label in label_set:
                count = 0
                for id_num, ordered_labels in ranked_labels:
                    if label == ordered_labels[N][0]:
                        count += 1
                label_counts.append(count)
            f.write('TOPIC COUNTS IN RANK %d\n' % (N+1))
            f.write('{:>25} {:>15}\n'.format('Topic', '# of Reports'))
            f.write('{:>25} {:>15}\n'.format('-----', '------------'))
            ordered_label_counts = sorted(zip(label_set, label_counts),
                                          key=lambda t: t[1], reverse=True)
            for label, count in ordered_label_counts:
                f.write('{:>25} {:>15}\n'.format(label, count))
            f.write('\n')

    # Write Top N terms for topics
    N = 50
    with open(output_folder + '/TOPICS_TOP_' + str(N) + '.csv', 'w') as f:
        for topic_label in topics:
            topic = topics[topic_label]
            topic_count = 0
            for term in topic:
                topic_count += topic[term]
            topic_data = [(term, 100*topic[term]/topic_count)
                          for term in topic]
            ordered_topic = sorted(topic_data, key=lambda t: t[1],
                                   reverse=True)
            f.write('{0},\n'.format(topic_label.upper()))
            f.write('{0},{1}\n'.format('Term', 'Probability'))
            for i in range(N):
                term = ordered_topic[i][0]
                prob = round(ordered_topic[i][1], 2)
                f.write('{0},{1}%\n'.format(term, prob))
            f.write(',\n,\n')


def write_document_topic_tags(document_filepaths, ranked_labels,
                              output_folder, top_n):
    # Document names should correspond 1-to-1 with ranked_labels
    document_names = [ntpath.basename(filepath)
                      for filepath in document_filepaths]
    for id_num, ordered_labels in ranked_labels:
        doc_name = document_names[id_num]
        output_file = output_folder + doc_name + '.tags'
        with open(output_file, 'w') as f:
            for i in range(top_n):
                label = ordered_labels[i][0]
                probability = ordered_labels[i][1]
                line = '{0},{1}\n'.format(label, probability)
                f.write(line)


def iterative_llda():
    pass


def run_for_scientific_articles():
    # Set up file paths and names
    articles_file = '../sciencewise/scientific_articles_17k.pickle'
    kpex_file = '../sciencewise/kpex_data_17k.pickle'
    articles_file = '../sciencewise/scientific_articles_3.5k.pickle'
    kpex_file = '../sciencewise/kpex_data_3.5k.pickle'
    # taxonomy_file = '../knowledge_base/twitter_ontology.csv'
    taxonomy_file = '../knowledge_base/sciencewise_concepts_27-may.csv'
    label_taxonomy_file = '../knowledge_base/twitter_ontology.csv'
    kpex_variants_file = None
    training_file = '../work/TRAIN.llda'
    testing_file = '../work/TEST.llda'
    tmt_file = 'tmt-0.4.0.jar'
    llda_learn_script = '6-llda-learn.scala'
    llda_infer_script = '7b-lda-infer.scala'
    model_path = '../work/model/'
    inference_file = '../work/inferences.tsv'
    work_folder = '../work/'

    # Load scientific articles and KPEX data
    with open(kpex_file, 'r') as f:
        kpex_data = pickle.load(f)
    with open(articles_file) as f:
        scientific_articles = pickle.load(f)

    # Prepare taxonomy and corpus
    taxonomy = prepare_taxonomy(taxonomy_file, cluster=False)
    corpus, underscores = process_scientific_articles(scientific_articles,
                                                      kpex_data,
                                                      taxonomy,
                                                      threshold=3)
    with open(work_folder + 'corpus', 'w') as f:
        pickle.dump(corpus, f)
    with open(work_folder + 'taxonomy', 'w') as f:
        pickle.dump(taxonomy, f)
    with open(work_folder + 'underscores', 'w') as f:
        pickle.dump(topics, f)

    # Get label set and label assignments
    label_taxonomy = prepare_taxonomy(label_taxonomy_file, cluster=True)
    labelset = create_labelset(label_taxonomy, [], corpus,
                               threshold=0)
    labels = assign_labels(labelset, label_taxonomy, corpus)
    with open(work_folder + 'label_taxonomy', 'w') as f:
        pickle.dump(topics, f)
    with open(work_folder + 'labelset', 'w') as f:
        pickle.dump(labelset, f)
    with open(work_folder + 'labels', 'w') as f:
        pickle.dump(labels, f)

    # Train LLDA
    write_data_set(corpus, labels, training_file, semisupervised=False)
    llda_learn(tmt_file, llda_learn_script, training_file, model_path)

    # Infer over all documents
    write_data_set(corpus, labels, testing_file, semisupervised=True)
    llda_infer(tmt_file, llda_infer_script, model_path, testing_file,
               inference_file)
    topics, ranked_labels = process_inference_results(model_path,
                                                      inference_file,
                                                      underscores)
    write_inference_results(topics, ranked_labels, work_folder)

    # Save inference data
    with open(work_folder + 'topics', 'w') as f:
        pickle.dump(topics, f)
    with open(work_folder + 'ranked_labels', 'w') as f:
        pickle.dump(topics, f)


def run_for_enb_reports():
    enb_file = '../enb/sw_enb_reports.csv'
    # taxonomy_file = '../knowledge_base/twitter_ontology.csv'
    taxonomy_file = '../knowledge_base/sciencewise_concepts_27-may.csv'
    label_taxonomy_file = '../knowledge_base/twitter_ontology.csv'
    kpex_concepts_file = '../enb/enb_corpus_kpex.kpex_n9999.txt'
    kpex_variants_file = None
    training_file = '../work/TRAIN'
    testing_file = '../work/TEST'
    tmt_file = 'tmt-0.4.0.jar'
    llda_learn_script = '6-llda-learn.scala'
    llda_infer_script = '7b-lda-infer.scala'
    model_path = '../work/llda_model/'
    inference_file = '../work/llda_inferences.tsv'
    work_folder = '../work/'

    # Prepare data for LLDA
    taxonomy = prepare_taxonomy(taxonomy_file, cluster=True)
    frequencies, synonyms = process_kpex_concepts(kpex_concepts_file,
                                                  kpex_variants_file, taxonomy)
    corpus, underscores = process_enb_reports(enb_file, synonyms, frequencies,
                                         taxonomy, threshold=10)
    label_taxonomy = prepare_taxonomy(label_taxonomy_file, cluster=True)
    labelset = create_labelset(label_taxonomy, frequencies, corpus,
                               threshold=10)
    labels = assign_labels(labelset, label_taxonomy, corpus)

    # Train LLDA
    write_data_set(corpus, labels, training_file, semisupervised=False)
    llda_learn(tmt_file, llda_learn_script, training_file, model_path)

    # Infer over all documents
    write_data_set(corpus, labels, testing_file, semisupervised=True)
    llda_infer(tmt_file, llda_infer_script, model_path, testing_file,
               inference_file)
    topics, ranked_labels = process_inference_results(model_path,
                                                      inference_file,
                                                      underscores)
    write_inference_results(topics, ranked_labels, work_folder)

    # Save all working data
    with open(work_folder + 'corpus', 'w') as f:
        pickle.dump(corpus, f)
    with open(work_folder + 'labelset', 'w') as f:
        pickle.dump(labelset, f)
    with open(work_folder + 'labels', 'w') as f:
        pickle.dump(labels, f)
    with open(work_folder + 'taxonomy', 'w') as f:
        pickle.dump(taxonomy, f)
    with open(work_folder + 'frequencies', 'w') as f:
        pickle.dump(frequencies, f)
    with open(work_folder + 'synonyms', 'w') as f:
        pickle.dump(synonyms, f)
    with open(work_folder + 'topics', 'w') as f:
        pickle.dump(topics, f)
    with open(work_folder + 'ranked_labels', 'w') as f:
        pickle.dump(topics, f)
    with open(work_folder + 'underscores', 'w') as f:
        pickle.dump(topics, f)
    with open(work_folder + 'label_taxonomy', 'w') as f:
        pickle.dump(topics, f)


def main():
    pass

if __name__ == '__main__':
    main(sys.argv)
