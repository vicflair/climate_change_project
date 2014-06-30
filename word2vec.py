# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:39:52 2014

@author: victor
"""

import csv
import multiprocessing
import os
import pickle
import re
import subprocess
import sys
from functools import partial
from itertools import izip
from nltk.tokenize import sent_tokenize, word_tokenize


def unpack_f(zipped_arguments, fcn):
    """ Intended to unpack zipped arguments, not necessarily iterable, to
    convert a f([x, y, z]) call to the f(x, y, z). This allows calls to
    multiprocessing.Pool().map() to take multiple iterable arguments.
    """
    return fcn(*zipped_arguments)


def filter_characters(text):
    # Keep only letters, hyphen, and underscores. Replace rest with space
    keep_chars = re.sub('[^a-zA-Z\-_]', ' ', text)
    # Condense spaces
    filtered_text = re.sub(' +', ' ', keep_chars)
    return filtered_text


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


def sanitize_article(article, kpex, taxonomy, threshold=0):
    # "Unpack" and rename KPEX data items
    frequencies = kpex[0]
    synonyms = kpex[1]
    # Convert to lowercase and filter "bad" characters
    article = filter_characters(article.lower())
    # Build taxonomy
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
    return article


def process_scientific_articles(scientific_articles, kpex_data, taxonomy,
                                threshold=0):
    """ Very similar to the process_enb_reports() except it takes kpex info
    in a different format since there is one KPEX file for every scientific
    article as compared to one KPEX file for all ENB reports.

    This function is separate because it may be later than the scientific
    articles require a different set of pre-processing steps.

    This function is processed in parallel.
    """
    # Sanitize scientific articles and filter with KPEX data, performing
    # processing in parallel
    par_sanitize = partial(sanitize_article, taxonomy=taxonomy,
                           threshold=threshold)
    sanitize = partial(unpack_f, fcn=par_sanitize)
    p = multiprocessing.Pool(4)
    corpus = p.map(sanitize, izip(scientific_articles, kpex_data))
    p.close()
    p.join()
    # Save correspondences between underscored and concatenated version of all
    # # n-grams in the corpus so we can do backwards conversion after LDA
    # all_terms = set(reduce(list.__add__, corpus))
    # underscores = {}
    # for term in all_terms:
    #     no_underscores = term.replace('_', '')
    #     underscores[no_underscores] = term
    return corpus


# Using a trained word vector model from Google's word2vec toolkit this
# gets a ranked list of matches over all terms associated with a concept.
# The concept should be provided as a list of terms, while the vector
# model should be provided as a filepath.
def get_concept_matches(concept, vector_model):
    # Collect word matches and their similarity scores for all terms in
    # the concept
    results = []
    for term in concept:
        command = ['./my_distance', vector_model, term]
        print ' '.join(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        text = p.communicate()[0]
        file_name = 'matches/' + term.replace(' ', '_') + '.txt'
        with open(file_name, 'w') as f:
            f.write(text)
        with open(file_name) as f:
            matches = f.readlines()
            for line in matches:
                similarity = re.search("0\.[0-9]+", line)
                word = re.search("[A-z]+", line)
                if similarity is not None:
                    results.append((word.group(0),
                                    float(similarity.group(0))))
    # Sort results by similarity score (highest first) and print to file
    results.sort(key=lambda tup: tup[1], reverse=True)
    concept_fname = ('concept_matches/' +
        concept[0].replace(' ', '_') + '.txt')
    with open(concept_fname, 'w') as f:
        for result in results:
            set_line = '{:>20} {:>20}'.format(result[0], str(result[1]))
            f.write(set_line + '\n')


def main():
    # Important files
    articles_file = '../sciencewise/scientific_articles_21k.pickle'
    kpex_file = '../sciencewise/kpex_data_21k.pickle'
    taxonomy_file = '../knowledge_base/sciencewise_concepts_27-may.csv'
    label_taxonomy_file = '../knowledge_base/twitter_ontology.csv'
    corpus_file = '../work/corpus.pickle'
    word2vec_corpus_file = '../work/word2vec_corpus.txt'
    vector_model_file = '../work/vectors.bin'
    # Load scientific articles and KPEX data
    with open(kpex_file, 'r') as f:
        kpex_data = pickle.load(f)
    with open(articles_file) as f:
        scientific_articles = pickle.load(f)
    # Create corpus for word2vec
    taxonomy = prepare_taxonomy(taxonomy_file, cluster=False)
    corpus = process_scientific_articles(scientific_articles,
                                         kpex_data, taxonomy,
                                         threshold=3)
    with open(corpus_file, 'w') as f:
        pickle.dump(corpus, f)
    all_words = ' '.join(corpus)
    with open(word2vec_corpus_file, 'w') as f:
        f.write(all_words)

    # Run word2vec
    command = ['./word2vec',
               '-train', word2vec_corpus_file,
               '-output', vector_model_file,
               '-cbow', '0',
               '-size', '200',
               '-window', '5',
               '-negative', '0',
               '-hs', '1',
               '-sample', '1e-3',
               '-threads', '12',
               '-binary', '1']

    print os.getcwd()

    # load climate change concept terms
    concepts = prepare_taxonomy(label_taxonomy_file, cluster=True)
    concepts = [concept for concept in concepts]
    for concept in concepts:
        get_concept_matches(concept, vector_model_file)


if __name__ == "__main__":
    main()

