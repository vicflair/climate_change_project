import csv
import json
import pickle
import re
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from functools import partial
from nltk.tokenize import sent_tokenize


# Allow larger entries in CSV files, e.g. for scientific articles
csv.field_size_limit(sys.maxsize)


def regex_search(term):
    # Search term is taxonomic term without underscores, optionally with a
    # terminal 's' (for plurals), and is case insensitive
    search_term = re.compile(r'(\b)' + term.replace('_', ' ') + r'(\b|s\b)',
                             re.I)
    return search_term


def capitalize_text(text, words_list, marker=''):
    """ Capitalize and mark, i.e. surround with special characters as
    identifiers, all taxonomic terms in a text, using a regular expressions for
    search.
    """
    for word in words_list:
        text = capitalize_term_re(text, word, marker)
    return text


def capitalize_term_re(text, term, marker=''):
    """ Returns text with input term capitalized and marked, e.g. if term =
    'flower' and marker = '$', then:
    'Roses are the best flowers.' --> 'Roses are the best $FLOWERS$s.'
    """
    # TODO: Copy this search method to concept/topic detectoin functions
    assert type(marker) is str
    search_term = regex_search(term)
    # Marked term is capitalized taxonomic term with 's' afterwards if plural
    marked_term = r'\1' + marker + term.upper() + marker + r'\2'
    annotated_text = re.sub(search_term, marked_term, text)
    return annotated_text


def write_annotated_docs(doc_file, concepts_file, topics_file,
                         marker=''):
    # FIXME: some terms are over marked, ****** instead of just ***
    # Capitalize taxonomic concept occurrences in original texts
    concept_taxonomy = load_taxonomy(concepts_file)
    topic_taxonomy = load_taxonomy(topics_file)
    with open(doc_file, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        docs = [row[7] for row in data]
    topics = reduce(list.__add__, [topic_taxonomy[topic]
                                   for topic in topic_taxonomy])
    concepts = reduce(list.__add__, [concept_taxonomy[concept]
                                     for concept in concept_taxonomy])
    all_key_words = list(set(topics + concepts))
    cap_words = partial(capitalize_text, words_list=all_key_words,
                           marker=marker)
    annotated_docs = map(cap_words, docs)
    # Write to file
    with open('../work/annotated_docs.tsv', 'w') as f:
        f.write('ID#\tANNOTATED TEXT\n')
        for i, doc in enumerate(annotated_docs):
            f.write('{0}\t{1}\n'.format(i, doc))
    return annotated_docs


def load_corpus(filename):
    if filename.endswith('.csv'):
        with open(filename, 'r') as f:
            data = csv.reader(f, delimiter='\t')
            corpus = [row[7] for row in data]
    elif filename.endswith('.pickle'):
        with open(filename, 'r') as f:
            corpus = pickle.load(f)
    return corpus


def load_taxonomy(filename):
    with open(filename, 'r') as f:
        data = csv.reader(f, delimiter=',')
        concepts = [filter(lambda x: x != '', row) for row in data]
    # Build concept taxonomy
    taxonomy = {}
    for concept in concepts:
        concept_name = concept[0].replace(' ', '_')
        terms = map(lambda s: s.lower().replace(' ', '_'), concept)
        taxonomy[concept_name] = list(set(terms))
    return taxonomy


def get_weight(sections, term_pos):
    """ Get weight for a particular occurrence of a term in a text:
     Title: 8x
     Abstract: 4x
     H1: 2x
     H2: 2x
     H3: 2x
     Conclusion: 2x
     Body: 2x
     Caption: 1x
     Bibliography: 1x

     sections is the indices of all 4 sections, in aforementioned order
     term_pos is the index of the term's position in the text string.

     We check, in reverse order (bibliography to title) whether term_pos >
     section_pos. The first section for which this is true is the section to
     which the term belongs.
    """
    # Set weights for each section
    weights = [1, 1, 2, 2, 2, 2, 2, 4, 8]
    # Working from end of text, determine in which section the term lies, and
    # apply weight accordingly
    weight = 1
    for i, section in enumerate(sections):
        if (term_pos > section) and (section != -1):
            weight = weights[i]
            break
    return weight


def find_sections(doc):
    # Find section headers, to use as reference for weighting
    title = doc.find('::TITLE::')
    abstract = doc.find('::ABSTRACT::')
    h1 = doc.find('::H1::')
    h2 = doc.find('::H2::')
    h3 = doc.find('::H3::')
    conclusion = doc.find('::CONCLUSION::')
    body = doc.find('::BODY::')
    caption = doc.find('::CAPTION::')
    bibliography = doc.find('::BIBLIOGRAPHY::')
    # Indices of section headers, in reverse order (according to XML dump)
    sections = [bibliography, caption, body, conclusion, h3, h2, h1, abstract,
                title]
    return sections


def detect_sent_concepts(doc, taxonomy):
    # Get section header locations
    sections = find_sections(doc)
    # Build list of all taxonomic terms
    terms = []
    for concept in taxonomy:
        terms += taxonomy[concept]
    # Detect sentence-specific concepts, and track position in document
    doc_pos = 0
    doc_concepts = []
    for sent in sent_tokenize(doc):
        sent_concepts = []
        for term in terms:
            search_term = regex_search(term)
            match_pos = [m.start() for m in re.finditer(search_term, sent)]
            if match_pos:
                # Assume only one match is valid per sentence.
                term_pos = match_pos[0] + doc_pos
                concept = term_to_concept(term, taxonomy)
                weight = get_weight(sections, term_pos)
                sent_concepts.append((concept, weight))
        # Add tuple of unique concepts, i.e. no repeats even if different terms
        # referring to the same concept appear
        if sent_concepts:
            doc_concepts.append(list(set(sent_concepts)))
        # Increment document position after processing each sentence
        doc_pos += len(sent)
    return doc_concepts


def detect_corpus_concepts(corpus, taxonomy):
    """ Get corpus which is nested list of concepts, representing documents
    and sentences.
    """
    # Sanitize
    par_detect = partial(detect_sent_concepts, taxonomy=taxonomy)
    p = Pool(4)
    corpus_concepts = p.map(par_detect, corpus)
    p.close()
    p.join()
    return corpus_concepts


def term_to_concept(term, taxonomy):
    concept_name = [concept for concept in taxonomy
                    if term in taxonomy[concept]]
    return concept_name[0]


def combine_weights(uncombined_items):
    combined_items = []
    for weighted_item in uncombined_items:
        # Get weight of every item which matches, or 0 for mismatch
        weights = map(lambda x: x[1] if x[0] == weighted_item[0] else 0,
                      uncombined_items)
        total_weight = sum(weights)
        # Added item to list of items with combined weights
        combined_item = (weighted_item[0], total_weight)
        combined_items.append(combined_item)
    # Remove duplicates, so each one represents the combined weight
    combined_items = list(set(combined_items))
    return combined_items


def valid_pairs(sent):
    valid_concept_pairs = []
    concepts = [weighted_concept[0] for weighted_concept in sent]
    score = [weighted_concept[1] for weighted_concept in sent][0]
    valid_issues = [
        ['climate_change', 'species_extinction'],
        ['climate_change', 'sea_level_rise'],
        ['climate_change', 'poverty'],
        ['climate_change', 'weather_extremes'],
        ['GHG_emissions', 'climate_change'],
        ['fossil_fuels', 'GHG_emissions'],
        ['deforestation', 'GHG_emissions'],
        ['transportation', 'GHG_emissions'],
        ['Clean_Development_Mechanism', 'GHG_emissions'],
        ['compliance_enforcement', 'GHG_emissions']
    ]
    for issue in valid_issues:
        if set(issue).issubset(concepts):
            valid_concept_pair = [(issue[0], issue[1]), score]
            valid_concept_pairs.append(valid_concept_pair)
    return valid_concept_pairs


def detect_sent_concept_pairs(doc):
    """ Given the set of detected concepts in a document's sentences,
    determine all sentence-specific concept pairs.
    """
    sent_pairs = []
    for sent in doc:
        # Get only valid issue concept pairs
        combos = valid_pairs(sent)
        sent_pairs.append(combos)
        # # Get n! pairs for all unique concepts if more than 2 concepts
        # combos = list(combinations(sent, 2))
        # # Transform concept combos representation from ((x, w), (y, w)) to the
        # # ((x, y), w) pair form, where x,y are concepts, and w is the weight
        # pairs = []
        # for combo in combos:
        #     # Sort pair so that (x,y) and (y,x) are a single form: (x,y)
        #     sorted_pair = tuple({combo[0][0], combo[1][0]})
        #     weight = combo[0][1]
        #     pair = (sorted_pair, weight)
        #     pairs.append(pair)
        # sent_pairs.append(pairs)
    if sent_pairs:
        sent_pairs = list(reduce(list.__add__, sent_pairs))
    # Combine weights of all instances of the same concept pair
    combined_weight_pairs = combine_weights(sent_pairs)
    return combined_weight_pairs


def get_concept_occurrences(corpus_file, concepts_file):
    """ Manual concept vector detection
    """
    # Load corpus and concept vectors
    concept_taxonomy = load_taxonomy(concepts_file)
    corpus = load_corpus(corpus_file)
    # Process corpus and return only concept terms on a per-document-sentence
    # level
    doc_concepts = detect_corpus_concepts(corpus, concept_taxonomy)
    # Get all sentence-level concept pairs for each document
    p = Pool(4)
    doc_pairs = map(detect_sent_concept_pairs, doc_concepts)
    p.close()
    p.join()
    # Get sentence-level pair-wise occurrence matrix
    sent_pairs = reduce(list.__add__, doc_pairs)
    pair_counts = combine_weights(sent_pairs)
    concepts = [c for c in concept_taxonomy]
    pair_matrix = []
    for concept1 in concepts:
        row = []
        for concept2 in concepts:
            pair_count = 0
            for count in pair_counts:
                if (concept1 in count[0]) and (concept2 in count[0]):
                    pair_count += count[1]
            row.append(pair_count)
        pair_matrix.append(row)
    # Write results
    write_concept_results(doc_pairs, pair_matrix, concept_taxonomy)
    # Save results
    with open('../work/doc_pairs.pickle', 'w') as f:
        pickle.dump(doc_pairs, f)
    # Save results
    with open('../work/pair_matrix.pickle', 'w') as f:
        pickle.dump(pair_matrix, f)
    # Save results
    with open('../work/concept_taxonomy.pickle', 'w') as f:
        pickle.dump(concept_taxonomy, f)

    return doc_pairs, pair_matrix


def write_concept_results(doc_concept_pairs, pair_matrix, concept_taxonomy):
    # Write co-occurrence matrix in csv format
    matrix_size = len(pair_matrix)
    concepts_list = [concept for concept in concept_taxonomy]
    matrix_file = '../work/manual_pairwise_concept_matrix.csv'
    with open(matrix_file, 'w') as f:
        f.write(','+','.join(concepts_list)+'\n')
        for i in range(matrix_size):
            line = '{:>30},' + '{:>10},'*matrix_size
            counts = map(str, pair_matrix[i])
            f.write(line.format(concepts_list[i], *counts)+'\n')
        print 'Wrote to ' + matrix_file
    # Write all extracted concept pairs for every document
    pairs_file = '../work/manual_concept_pairs_per_doc.tsv'
    with open(pairs_file, 'w') as f:
        f.write('ID\tCONCEPT PAIRS\n')
        for i, doc in enumerate(doc_concept_pairs):
            line = '{}\t' + ' '.join(['{}']*len(doc)) + '\n'
            f.write(line.format(i, *doc))
        print 'Wrote to ' + pairs_file


def get_topic_distributions(corpus_file, topics_file):
    """ Manual topic vector detection
    """
    # Get topic vectors and corpus
    topic_taxonomy = load_taxonomy(topics_file)
    corpus = load_corpus(corpus_file)
    sent_topics = detect_corpus_concepts(corpus, topic_taxonomy)
    doc_topics = map(lambda x: reduce(list.__add__, x) if x else [],
                     sent_topics)
    # Get ranked total weighted topic counts for each document
    total_doc_topics = map(combine_weights, doc_topics)
    par_sort = partial(sorted, key=lambda x: x[1], reverse=True)
    ranked_doc_topics = map(par_sort, total_doc_topics)
    # Get topic proportions for each document
    topic_freq = []
    for ranked_doc in ranked_doc_topics:
        if ranked_doc:
            total = sum(map(lambda x: x[1], ranked_doc))
            freq = []
            for topic in ranked_doc:
                percentage = round(100.0 * topic[1]/total, 1)
                freq.append((topic[0], percentage))
            topic_freq.append(freq)
        else:
            topic_freq.append([])
    # Write results
    write_topic_results(ranked_doc_topics, topic_freq)
    # Save results to pickle
    with open('../work/ranked_doc_topics.pickle', 'w') as f:
        pickle.dump(ranked_doc_topics, f)
    with open('../work/topic_freq.pickle', 'w') as f:
        pickle.dump(topic_freq, f)
    return ranked_doc_topics, topic_freq


def write_topic_results(ranked_doc_topics, topic_freq):
    # Write top 3 topics with their weighted counts
    max_n = 3
    top_3_counts_file = '../work/manual_top_3_topic_counts.tsv'
    with open(top_3_counts_file, 'w') as f:
        f.write('ID\tTOP 3 TOPICS\n')
        for i, ranked_topics in enumerate(ranked_doc_topics):
            n = min(max_n, len(ranked_topics))
            line = '{}\t'.format(i)
            for j in range(n):
                line += ' {}({})'.format(*ranked_topics[j])
            line += '\n'
            f.write(line)
        print 'Wrote to ' + top_3_counts_file
    # Write top 3 topics with their frequency based on weighted counts
    top_3_freqs_file = '../work/manual_top_3_topic_freqs.tsv'
    with open(top_3_freqs_file, 'w') as f:
        f.write('ID\tTOP 3 TOPICS\n')
        for i, ranked_topics in enumerate(topic_freq):
            n = min(max_n, len(ranked_topics))
            line = '{}\t'.format(i)
            for j in range(n):
                line += ' {}({}%)'.format(*ranked_topics[j])
            line += '\n'
            f.write(line)
        print 'Wrote to ' + top_3_freqs_file


def write_json(tsv_file):
    # TODO: write a function to combine results data structures so that we
    # don't need to read from TSV
    # Fields to parse and expand, known a priori
    counts = 'TOP 3 TOPICS (WEIGHTED COUNTS)'
    freqs = 'TOP 3 TOPICS (WEIGHTED FREQUENCY)'
    pairs = 'CONCEPT PAIRS'
    # Split pattern for concept pairs: split on space, if between ) and (
    regex_pair = re.compile(('(?<=\)) (?=\()'))
    # Split pattern for each item (concept 1, concept 2, #) in pair
    regex_items = re.compile('([\w_]+)[^\w]+([\w_]+)[^\d]+([\d]+)')
    # split pattern for topic counts and freqs
    regex_counts = re.compile('([\w]+)\(([\d]+)')
    # Load TSV file as dict
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        data = [row for row in reader]
    # Parse and expand topics and concepts as a list
    for i, datum in enumerate(data):
        # TODO: find out why ID is sometime missing?
        # Turn ID to int from string, sometimes ID is missing
        if data[i]['ID#']:
            data[i]['ID#'] = int(data[i]['ID#'])
        else:
            data[i]['ID#'] = i
        # parse topic counts
        topic_counts = datum[counts].split()
        data[i][counts] = {}
        for t_count in topic_counts:
            items = re.search(regex_counts, t_count)
            topic = items.groups()[0]
            # TODO: Account for TSV reading error
            if topic == '_Adaptation':
                topic = 'Vulnerability_&_Adaptation'
            topic_count = int(items.groups()[1])
            data[i][counts][topic] = topic_count
        # parse topic frequencies
        topic_freqs = datum[freqs].split()
        data[i][freqs] = {}
        for t_freq in topic_freqs:
            items = re.search(regex_counts, t_freq)
            topic = items.groups()[0]
            # TODO: Account for TSV reading error
            if topic == '_Adaptation':
                topic = 'Vulnerability_&_Adaptation'
            topic_freq = int(items.groups()[1])
            data[i][freqs][topic] = topic_freq
        # if pairs exist, parse separate concept pairs
        if datum[pairs]:
            concept_pairs = re.split(regex_pair, datum[pairs])
            data[i][pairs] = []
        else:
            concept_pairs = []
        for c_pair in concept_pairs:
            # TODO: Find source of "self-pair" error
            # if erroneous "self-pair", e.g. "(('GHG_emissions', ), 2)", OMIT!
            if '\',)' in c_pair:
                pass
            # if not, go ahead and parse
            else:
                # parse, reorder, & replace in results data structure
                items = re.search(regex_items, c_pair)
                weighted_count = int(items.groups()[2])
                concept1 = items.groups()[0]
                concept2 = items.groups()[1]
                new_pair = [weighted_count, concept1, concept2]
                data[i][pairs].append(new_pair)
        # if applicable, sort
        data[i][pairs] = sorted(data[i][pairs], key=lambda x: x[0],
                                reverse=True)
    # Save updated JSON data in indented form
    json_out_file = '../work/manual_detection_INDENT.json'
    with open(json_out_file, 'w') as f:
        json.dump(data, f, indent=4)
        print 'Wrote to ' + json_out_file
    json_out_file = '../work/manual_detection_LINE.json'
    with open(json_out_file, 'w') as f:
        f.write('[\n')
        for datum in data:
            json.dump(datum, f)
            f.write(',\n')
        f.write(']')
        print 'Wrote to ' + json_out_file


def main():
    enb_file = '../enb/sw_enb_reports.csv'
    swa_file = '../sciencewise/scientific_articles_100.pickle'
    concepts_file = '../knowledge_base/manual_concept_vectors.csv'
    topics_file = '../knowledge_base/manual_topic_vectors.csv'
    issues_file = '../knowledge_base/manual_issues.csv'

    start = time.time()
    output = get_concept_occurrences(enb_file, concepts_file)
    print time.time() - start

    start = time.time()
    output = get_topic_distributions(enb_file, concepts_file)
    print time.time() - start

    with open(issues_file, 'r') as f:
        data = csv.reader(f, delimiter=',')
        issues = [tuple(row) for row in data]


if __name__ == '__main__':
    main()
