import csv
import collections
import string
from nltk.tokenize import sent_tokenize, word_tokenize



# Get ENB corpus
enb_file = '../enb/ENB_Reports.csv'
enb = []
with open(enb_file, 'r') as f:
    data = csv.reader(f, delimiter='\t')
    enb = [row[7] for row in data]

# Get relations
# TODO: Increase number of synonyms and antonyms in relations list
relations_file = 'manual_relations_list.csv'
relations = []
with open(relations_file, 'r') as f:
    data = csv.reader(f, delimiter='\t')
    relations = [row for row in data]
    if relations[0][1] is not 'synonym' or 'antonym':
        relations.pop(0)
rel = {}
for i in relations:
    main_verb = i[0]
    nym = i[1]
    related_verb = i[2]
    if main_verb not in rel:
        # Initialize main_verb in relations structure before adding
        # related_verb. Then add related_verb to the appropriate (synonym or
        # antonym) list. It should be same as main_verb for 1st occurrence.
        rel[main_verb] = dict()
        rel[main_verb]['synonym'] = []
        rel[main_verb]['antonym'] = []
    # Append new synonym or antonym.
    rel[main_verb][nym].append(related_verb)
relations = rel

# Get known issues
issues_file = 'known_issues_subset.csv'
known_issues = []
with open(issues_file, 'r') as f:
    data = csv.reader(f, delimiter=',')
    known_issues = [row for row in data]

# From known issues, get concepts
concepts = []
for issue in known_issues:
    left_concept = issue[0].rstrip()
    right_concept = issue[2].rstrip()
    concepts += [left_concept, right_concept]
concepts = list(set(concepts))

# Find all occurrences of concepts
concept_occurrences = []
for report in enb:
    detected_concepts = [concept for concept in concepts if concept
                         in report.lower()]
    concept_occurrences.append(detected_concepts)

# Find all occurrences of relations
relation_occurrences = []
for report in enb:
    detected_occurrences = []
    for main_verb in relations:
        detected_synonyms = [verb for verb in relations[main_verb]['synonym']
                             if verb in report.lower()]
        detected_antonyms = [verb for verb in relations[main_verb]['antonym']
                             if verb in report.lower()]
        if detected_synonyms:
            detected_occurrences.append(main_verb)
        if detected_antonyms:
            detected_occurrences.append('not ' + main_verb)
    relation_occurrences.append(detected_occurrences)

# Find all occurrences of issues based on joint occurrence of concepts without
# regard for placement. There are no instances where a single report has the
# two concepts and one verb in any known issue.
joint_occurrences = []
for relation, concept, report in zip(relation_occurrences, concept_occurrences, enb):
    potential_issue = tuple()
    if relation and len(concept) >= 2:
        potential_issue = (relation, concept, report)
    joint_occurrences.append(potential_issue)
issues = []
for co, report in zip(joint_occurrences, enb):
    if co:
        terms = co[1]
        verbs = co[0]
        for issue in known_issues:
            left_concept = issue[0]
            relation = issue[1]
            right_concept = issue[2]
            if (left_concept in terms) and (right_concept in terms):
                if relation in verbs:
                    potential_issue = (left_concept, relation, right_concept,
                                       report)
                    issues.append(potential_issue)
                if 'not ' + relation in verbs:
                    potential_issue = (left_concept, 'not ' + relation,
                                       right_concept, report)
                    issues.append(potential_issue)

# Print analysis statistics
unique_reports = set([issue[3] for issue in issues])
num_issues = len(issues)
num_issue_reports = len(unique_reports)
num_joint = len([co for co in joint_occurrences if co])
total_reports = len(enb)
total_concepts = len(reduce(list.__add__, [c for c in concept_occurrences if c]))
total_relations = len(reduce(list.__add__, [r for r in relation_occurrences if r]))
print ''
statement = '{0} potential issues found in {1} reports'
print statement.format(num_issues, num_issue_reports)
statement = '{num_co} reports with issue triples out of {num_reps} reports'
print statement.format(num_co=num_joint, num_reps= total_reports)
statement = '{0} concepts and {1} merged relations found in {2} reports.'
print statement.format(total_concepts, total_relations, total_reports)

# Check for repeated phrases, using:
# http://stackoverflow.com/questions/4526762/repeated-phrases-in-the-text-python
# FIXME: use instead https://github.com/raypereda/repeating-phrases
all_enb = ' '.join(enb)


def words(text):
    for sent in sent_tokenize(text):
        for word in word_tokenize(sent):
            if word[0] in string.ascii_letters and word != 'http':
                yield word


def phrases(words, length):
    phrase = []
    for word in words:
        phrase.append(word)
        if len(phrase) > length:
            phrase.remove(phrase[0])
        if len(phrase) == length:
            yield tuple(phrase)

# Get counts for phrases of size 10
#stuff = list(phrases(words(all_enb), 10))
counts = collections.defaultdict(int)
phrase_length = 20
for phrase in phrases(words(all_enb), phrase_length):
    counts[phrase] += 1


def repeated_phrases(counts, threshold):
    # FIXME: How to make sure overlapping phrases aren't counted?
    repeated = []
    for item in counts:
        if counts[item] == threshold:
            repeated.append((item, counts[item]))
    return repeated

for i in range(2, 8):
    num_repeats = len(repeated_phrases(counts, i))
    statement = '# of phrases of length {0} repeated {1} times: {2}'
    print statement.format(phrase_length, i, num_repeats)


