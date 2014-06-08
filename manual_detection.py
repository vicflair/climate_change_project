import csv
from llda_enb import remove_non_ascii, make_taxonomic_ngrams, lemmatize_word, \
    filter_terms
from nltk.tokenize import sent_tokenize, word_tokenize


def detect_enb_concepts(enb_file, taxonomy):
    """ Get corpus which is a list of sentences, where each sentence is
    represented as a list of words
    """
    corpus = []
    # Get ENB reports
    with open(enb_file, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        reports = [row[7].lower() for row in data]
    for report in reports:
        # Remove non-ASCII characters
        report = remove_non_ascii(report)
        # Chain uni-grams into n-grams
        assert type(taxonomy) is dict
        report = make_taxonomic_ngrams(taxonomy, report)
        # Separate into sentences and remove stop words, non-words, and
        # otherwise all words not in taxonomy
        doc = []
        for sent in sent_tokenize(report):
            sent_words = [word for word in word_tokenize(sent)]
            filtered_words = filter_terms(sent_words, taxonomy=taxonomy,
                                          frequencies=None, threshold=0)
            doc.append(filtered_words)
        corpus.append(doc)
    return corpus


def get_concept_occurrences(enb_file, concepts_file):
    """ Manual concept vector detection
    """
    # Get concept vectors
    with open(concepts_file, 'r') as f:
        data = csv.reader(f, delimiter=',')
        concepts = [filter(lambda x: x != '', row) for row in data]
    # Filter corpus for detected concept vectors
    concept_taxonomy = {}
    for concept in concepts:
        concept_name = concept[0].replace(' ', '_')
        terms = map(lambda s: s.lower().replace(' ', '_'), concept)
        concept_taxonomy[concept_name] = terms
    concept_corpus = detect_enb_concepts(enb_file, concept_taxonomy)
    concept_sentence_corpus = reduce(list.__add__, concept_corpus)
    # Get list of detected concepts for each document
    detected_concepts = [[concept for concept in concept_taxonomy
                          for term in terms
                          if term in concept_taxonomy[concept]]
                         for terms in concept_sentence_corpus]
    # Get sentence-level pair-wise occurrence matrix
    pair_freq = [[len([i for i, detected in enumerate(detected_concepts)
                       if concept1 in detected and concept2 in detected])
                  for concept1 in concept_taxonomy]
                 for concept2 in concept_taxonomy]
    # Write co-occurrence matrix in csv format
    matrix_size = len(pair_freq)
    concepts_list = [concept for concept in concept_taxonomy]
    with open('../work/manual_pairwise_concept_occurrence.csv', 'w') as f:
        f.write(','+','.join(concepts_list)+'\n')
        for i in range(matrix_size):
            line = '{:>30},' + '{:>10},'*matrix_size
            counts = map(str, pair_freq[i])
            f.write(line.format(concepts_list[i], *counts)+'\n')
    return detected_concepts


def get_topic_distributions(enb_file, topics_file):
    """ Manual topic vector detection
    """
    # Get topic vectors
    with open(topics_file, 'r') as f:
        data = csv.reader(f, delimiter=',')
        topics = [filter(lambda x: x != '', row) for row in data]
    # Filter corpus for detected topic vectors
    topic_taxonomy = {}
    for topic in topics:
        topic_name = topic[0].replace(' ', '_')
        terms = map(lambda s: s.lower().replace(' ', '_'), topic)
        topic_taxonomy[topic_name] = terms
    topic_sentence_corpus = detect_enb_concepts(enb_file, topic_taxonomy)
    topic_corpus = map(lambda x: reduce(list.__add__, x),
                       topic_sentence_corpus)
    # Get list of repeated topic occurrences for each document
    detected_topics = [[topic for topic in topic_taxonomy
                        for term in terms if term in topic_taxonomy[topic]]
                       for terms in topic_corpus]
    # Get topic proportions for each document
    topic_freq = [[detected.count(topic)/float(len(detected))
                   if len(detected) is not 0
                   else 0 for topic in topic_taxonomy]
                  for detected in detected_topics]
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
                ranked_topics = sorted(dist, key=lambda t:t[1], reverse=True)
                ranked_labels = [x[0] for x in ranked_topics]
                ranked_freq = [x[1] for x in ranked_topics]
                line = '{:>7}' + ',{:>30}'*len(ranked_topics) + '\n'
                f.write(line.format(i, *ranked_labels))
                f.write(line.format('', *ranked_freq))
    return detected_topics, topic_freq, topics_list


def main():
    enb_file = '../enb/sw_enb_reports.csv'
    concepts_file = '../knowledge_base/manual_concept_vectors.csv'
    topics_file = '../knowledge_base/manual_topic_vectors.csv'
    issues_file = '../knowledge_base/manual_issues.csv'

    detected_concepts = get_concept_occurrences(enb_file, concepts_file)
    output = get_topic_distributions(enb_file, topics_file)
    detected_topic = output[0]
    topic_freq = output[1]
    topics_list = output[2]

    with open(issues_file, 'r') as f:
        data = csv.reader(f, delimiter=',')
        issues = [tuple(row) for row in data]




