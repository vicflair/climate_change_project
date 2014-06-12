import csv
import re
import sys
import time
from itertools import combinations
from llda_enb import lemmatize_word, filter_terms, make_taxonomic_ngrams, \
    remove_non_ascii
from multiprocessing import Pool
from functools import partial
from nltk.tokenize import sent_tokenize, word_tokenize


# Allow larger entries in CSV files, e.g. for scientific articles
csv.field_size_limit(sys.maxsize)


def capitalize_text(text, taxonomy, marker=''):
    """ Capitalize and mark, i.e. surround with special characters as
    identifiers, all taxonomic terms in a text, using a regular expressions for
    search.
    """
    terms = []
    for concept in taxonomy:
        terms += taxonomy[concept]
    for term in terms:
        text = capitalize_term_re(text, term, marker)
    return text


def capitalize_term(text, term, marker=None):
    matches = [m.start() for m in
               re.finditer(' ' + term.replace('_', ' ') + ' ',
                           text.lower())]
    matches = sorted(matches, reverse=True)
    if marker:
        assert type(marker) is str
        for match in matches:
            marked_term = ' ' + marker + term.upper() + marker
            text = text[0:match] + marked_term + text[match+len(term)+1:]
    else:
        matches = map(lambda x: x+1, matches)
        for match in matches:
            text = text[0:match] + term.upper() + text[match+len(term):]
    return text


def capitalize_term_re(text, term, marker=''):
    """ Returns text with input term capitalized and marked, e.g. if term =
    'flower' and marker = '$', then:
    'Roses are the best flowers.' --> 'Roses are the best $FLOWERS$s.'
    """
    # TODO: Copy this search method to concept/topic detectoin functions
    assert type(marker) is str
    # Marked term is capitalized taxonomic term with 's' afterwards if plural
    marked_term = ' ' + marker + term.upper() + marker + r'\1'
    # Search term is taxonomic term without underscores, optionally with a
    # terminal 's' (for plurals), and is case insensitive
    search_term = re.compile(' ' + term.replace('_', ' ') + '( |s )', re.I)
    annotated_text = re.sub(search_term, marked_term, text)
    return annotated_text


def write_annotated_docs(doc_file, taxonomy_file, marker=''):
    # Capitalize taxonomic concept occurrences in original texts
    taxonomy = load_taxonomy(taxonomy_file)
    with open(doc_file, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        docs = [row[7] for row in data]
    cap_text = partial(capitalize_text, taxonomy=taxonomy, marker=marker)
    annotated_docs = map(cap_text, docs)
    # Write to file
    with open('../work/annotated_docs.csv', 'w') as f:
        f.write('ID#, ANNOTATED TEXT\n')
        for i, doc in enumerate(annotated_docs):
            f.write('{0},{1}\n'.format(i, doc))
    return annotated_docs


def load_taxonomy(filename):
    with open(filename, 'r') as f:
        data = csv.reader(f, delimiter=',')
        concepts = [filter(lambda x: x != '', row) for row in data]
    # Build concept taxonomy
    taxonomy = {}
    for concept in concepts:
        concept_name = concept[0].replace(' ', '_')
        terms = map(lambda s: s.lower().replace(' ', '_'), concept)
        taxonomy[concept_name] = terms
    return taxonomy


def sanitize_enb_doc(report, taxonomy):
    # Remove non-ASCII characters
    report = remove_non_ascii(report)
    # Chain uni-grams into n-grams
    assert type(taxonomy) is dict
    report = make_taxonomic_ngrams(taxonomy, report)
    # Separate into sentences and remove stop words, non-words, and
    # otherwise all words not in taxonomy
    # TODO: Need to optimize this segment
    doc = []
    for sent in sent_tokenize(report):
        sent_words = [lemmatize_word(word) for word in word_tokenize(sent)]
        filtered_words = filter_terms(sent_words, taxonomy=taxonomy,
                                      frequencies=None, threshold=0)
        doc.append(filtered_words)
    return doc


def detect_enb_concepts(enb_file, taxonomy):
    """ Get corpus which is a list of sentences, where each sentence is
    represented as a list of words
    """
    # Get ENB reports
    with open(enb_file, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        reports = [row[7].lower() for row in data]
    # Sanitize
    par_sanitize = partial(sanitize_enb_doc, taxonomy=taxonomy)
    p = Pool(4)
    corpus = p.map(par_sanitize, reports)
    p.close()
    p.join()
    return corpus


def term_to_concept(term, taxonomy):
    concept_name = [concept for concept in taxonomy
                    if term in taxonomy[concept]]
    return concept_name[0]


def get_concept_occurrences(enb_file, concepts_file):
    """ Manual concept vector detection
    """
    # Get concept vectors
    concept_taxonomy = load_taxonomy(concepts_file)
    # Process corpus and return only concept terms on a per-document-sentence
    # basis
    terms_corpus = detect_enb_concepts(enb_file, concept_taxonomy)
    # Transform concept-terms representation of corpus to pure concepts
    doc_concepts = []
    for doc in terms_corpus:
        sentences = []
        for sentence in doc:
            # Get all unique concepts in each sentence
            sent_concepts = set([term_to_concept(term, concept_taxonomy)
                                 for term in sentence])
            sentences.append(sent_concepts)
        doc_concepts.append(sentences)
    # Accumulate list of detected concepts for each sentence
    sent_concepts = reduce(list.__add__, doc_concepts)
    # Get all unique sentence-level concept pairs for each document
    doc_concept_pairs = []
    for doc in doc_concepts:
        sentences = []
        for sent in doc:
            sent_pairs = list(combinations(sent, 2))
            sentences.append(sent_pairs)
        # Don't repeat pairs, even if they occur multiple times in 1 doc
        unique_doc_pairs = tuple(set(reduce(list.__add__, sentences)))
        doc_concept_pairs.append(unique_doc_pairs)
    # Get sentence-level pair-wise occurrence matrix
    pair_freq = [[len([i for i, detected in enumerate(sent_concepts)
                       if concept1 in detected and concept2 in detected])
                  for concept1 in concept_taxonomy]
                 for concept2 in concept_taxonomy]
    write_concept_results(doc_concept_pairs, pair_freq, concept_taxonomy)
    return doc_concepts


def write_concept_results(doc_concept_pairs, pair_freq, concept_taxonomy):
    # Write co-occurrence matrix in csv format
    matrix_size = len(pair_freq)
    concepts_list = [concept for concept in concept_taxonomy]
    with open('../work/manual_pairwise_concept_occurrence.csv', 'w') as f:
        f.write(','+','.join(concepts_list)+'\n')
        for i in range(matrix_size):
            line = '{:>30},' + '{:>10},'*matrix_size
            counts = map(str, pair_freq[i])
            f.write(line.format(concepts_list[i], *counts)+'\n')
    # Write all extracted concept pairs for every document
    with open('../work/manual_concept_pairs_per_doc.tsv', 'w') as f:
        f.write('ID\tCONCEPT PAIRS\n')
        for i, doc in enumerate(doc_concept_pairs):
            line = '{}\t' + ' '.join(['{}']*len(doc)) + '\n'
            f.write(line.format(i, *doc))


def get_topic_distributions(enb_file, topics_file):
    """ Manual topic vector detection
    """
    # Get topic vectors
    topic_taxonomy = load_taxonomy(topics_file)
    topic_sentence_corpus = detect_enb_concepts(enb_file, topic_taxonomy)
    topic_corpus = map(lambda x: reduce(list.__add__, x),
                       topic_sentence_corpus)
    # Get list of repeated topic occurrences for each document
    detected_topics = [[topic for topic in topic_taxonomy
                        for term in terms if term in topic_taxonomy[topic]]
                       for terms in topic_corpus]
    # Get topic proportions for each document
    topic_freq = [[round(100 * detected.count(topic)/float(len(detected)), 1)
                   if len(detected) is not 0
                   else 0 for topic in topic_taxonomy]
                  for detected in detected_topics]
    write_topic_results(topic_freq, topic_taxonomy)
    return detected_topics, topic_freq, topic_taxonomy


def write_topic_results(topic_freq, topic_taxonomy):
    # Write per-document topic distributions for each document in csv format
    topics_list = [topic for topic in topic_taxonomy]
    n_topics = len(topics_list)
    with open('../work/manual_per-document_topic_distributions.csv', 'w') as f:
        f.write('ID,' + ','.join(topics_list) + '\n')
        for i, distribution in enumerate(topic_freq):
            line = '{:>5},'*(n_topics+1) + '\n'
            f.write(line.format(i, *topic_freq[i]))
    # Write topics in frequency order (except if freq = 0) only for documents
    # with detected topics
    with open('../work/manual_document_ranked_topics.csv', 'w') as f:
        f.write('ID, TOPICS\n')
        for i, distribution in enumerate(topic_freq):
            if sum(distribution) is not 0:
                dist = [x for x in zip(topics_list, distribution)
                        if x[1] != 0.0]
                ranked_topics = sorted(dist, key=lambda t: t[1], reverse=True)
                ranked_labels = [x[0] for x in ranked_topics]
                ranked_freq = [x[1] for x in ranked_topics]
                line = '{:>7}' + ',{:>30}'*len(ranked_topics) + '\n'
                f.write(line.format(i, *ranked_labels))
                f.write(line.format('', *ranked_freq))
    # Write top 3 topics in frequency order (except if freq = 0) only for
    # documents with detected topics
    with open('../work/manual_document_ranked_top_3_topics.csv', 'w') as f:
        f.write('ID, TOP 3 TOPICS\n')
        for i, distribution in enumerate(topic_freq):
            line = '{},'.format(i)
            if sum(distribution) is not 0:
                dist = [x for x in zip(topics_list, distribution)
                        if x[1] != 0.0]
                ranked_topics = sorted(dist, key=lambda t: t[1], reverse=True)
                for j in range(min(len(ranked_topics), 3)):
                    line += '{}({}%) '.format(*ranked_topics[j])
            f.write(line + '\n')


def main():
    enb_file = '../enb/sw_enb_reports.csv'
    concepts_file = '../knowledge_base/manual_concept_vectors.csv'
    topics_file = '../knowledge_base/manual_topic_vectors.csv'
    issues_file = '../knowledge_base/manual_issues.csv'

    output = get_concept_occurrences(enb_file, concepts_file)
    output = get_topic_distributions(enb_file, topics_file)

    with open(issues_file, 'r') as f:
        data = csv.reader(f, delimiter=',')
        issues = [tuple(row) for row in data]




