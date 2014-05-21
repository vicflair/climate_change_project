import pickle
import subprocess
from llda_enb import *


# Get corpus texts
from llda_enb import *
enb_file = '../enb/ENB_Reports.csv'
# taxonomy_file = '../enb/ENB_Issue_Dictionaries.csv'
# kpex_concepts_file = 'enb_corpus_kpex.kpex_n9999.txt'
# kpex_variants_file = 'KPEX_ENB_term_variants.txt'
# taxonomy = prepare_taxonomy(taxonomy_file, cluster=False)
# frequencies, synonyms = process_kpex_concepts(kpex_concepts_file,
#                                               kpex_variants_file, taxonomy)
# corpus = prepare_training_set(enb_file, synonyms, frequencies, taxonomy)
with open('taxonomy', 'r') as f:
    original_taxonomy = pickle.load(f)
    taxonomy = map(lambda s: s.replace('_', ''), original_taxonomy)
with open('enb_corpus', 'r') as f:
    corpus = pickle.load(f)
reports = map(lambda x: ' '.join(x), corpus)

# Get report metadata as non-overlapping facets
hard_facets = dict()
with open(enb_file, 'r') as f:
    data = csv.reader(f, delimiter='\t')
    hard_facets['city'] = [row[2] for row in data]
    hard_facets['title'] = [row[4] for row in data]
    hard_facets['country'] = [row[5] for row in data]
    hard_facets['year'] = [row[6] for row in data]

# Get overlapping facets:
soft_facets = dict()
with open('countries', 'r') as f:
    countries = pickle.load(f)
actors = []
for report in reports:
    actors.append([country.replace(' ', '_') for country in countries
                   if ' ' + country + ' ' in report])
soft_facets['actor'] = actors

# Run L-LDA
tmt_file = 'tmt-0.4.0.jar'
llda_script = '6-llda-learn.scala'
training_file = 'labeled_by_actors'
output_folder = 'topics_actors'
labels = soft_facets['actor']
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
    label_index = f.readlines()
    label_index = map(str.rstrip, label_index)
topics = dict()
with open(output_folder+'/01500/results.csv', 'r') as f:
    data = csv.reader(f, delimiter=',')
    data = [row for row in data]
    for i, label in enumerate(label_index):
        topics[label] = dict()
        for j, term in enumerate(terms):
            topics[label][term] = float(data[i][j])

# View top N concepts
N = 15
for label in label_index:
    print ''
    print '-'*17+label+'-'*17
    ordered = [(term, topics[label][term]) for term in terms
               if term in taxonomy]
    ordered = sorted(ordered, key=lambda tup: tup[1], reverse=True)
    new_ordered = []
    for item in ordered:
        full_term = [oterm for oterm in original_taxonomy
                     if oterm.replace('_', '') == item[0]]
        new_ordered.append((full_term[0], item[1]))
    for i in range(N):
        #print new_ordered[i][0], str(round(new_ordered[i][1],1)) + '%'
        line = '%40s %5s' % (new_ordered[i][0], str(round(new_ordered[i][1],1)) + '%')
        print line

with open('results_cluster_by_actors.txt', 'w') as f:
    N = 15
    for label in label_index:
        f.write('\n')
        f.write('-'*17+label+'-'*17+'\n')
        ordered = [(term, topics[label][term]) for term in terms
                   if term in taxonomy]
        ordered = sorted(ordered, key=lambda tup: tup[1], reverse=True)
        new_ordered = []
        for item in ordered:
            full_term = [oterm for oterm in original_taxonomy
                         if oterm.replace('_', '') == item[0]]
            new_ordered.append((full_term[0], item[1]))
        for i in range(N):
            #print new_ordered[i][0], str(round(new_ordered[i][1],1)) + '%'
            line = '%40s %5s' % (new_ordered[i][0], str(round(new_ordered[i][1],1)) + '%')
            f.write(line+'\n')

# Perform inference on unlabeled reports
llda_script = '7-llda-infer.scala'
output_file = 'inferences_by_actor.tsv'
target_file = 'unlabeled_documents'
print 'Writing target file with unlabeled documents.'
with open(target_file, 'w') as f:
    for i, label in enumerate(labels):
        if not label:
            text = '\"' + ' '.join(corpus[i]) + '\"'
            text = text.replace('_', '')
            labels_str = ''
            line = ','.join([str(i), labels_str, text]) + '\n'
            f.write(line)
print 'Performing inference on unlabeled documents.'
llda_infer(tmt_file, llda_script, output_folder, target_file, output_file)

# Get ranked labels for each report
with open(output_file, 'r') as f:
    data = csv.reader(f, delimiter=',')
    inferences = [(int(row[0]), map(float, row[1:])) for row in data]
ranked_labels = []
for inference in inferences:
    id_num = inference[0]
    label_distribution = zip(label_index, inference[1])
    ordered_labels = sorted(label_distribution, key=lambda t: t[1],
                                 reverse=True)
    ranked_labels.append((id_num, ordered_labels))

# Print top N labels for each report
N = 3
for id_num, ordered_labels in ranked_labels:
    print 'Report ID# %d' % id_num
    print '{:>20} {:>20}-'.format('--Actors--', '--Proportion-')
    for i in range(N):
        print '{:>20} {:>20}%'.format(ordered_labels[i][0],
                                      round(100 * ordered_labels[i][1], 2))
    print ''

# Print counts for # of documents with each label in the top N
N = 1
label_counts = []
for label in label_index:
    count = 0
    for id_num, ordered_labels in ranked_labels:
        if label in [pair[0] for pair in ordered_labels[:N]]:
            count += 1
    label_counts.append(count)
print 'TOP %d ACTOR COUNTS' % N
print '{:>20} {:>15}'.format('Actor', '# of Reports')
print '{:>20} {:>15}'.format('-----', '------------')
for label, count in zip(label_index, label_counts):
    print '{:>20} {:>15}'.format(label, count)
with open('TOP_1_ACTOR_COUNTS.txt', 'w') as f:
    f.write('ACTOR, # OF REPORTS\n')
    for label, count in zip(label_index, label_counts):
        f.write(label + ',' + str(count))

# Print counts for # of documents with each label in the top N
N = 3
label_counts = []
for label in label_index:
    count = 0
    for id_num, ordered_labels in ranked_labels:
        if label in [pair[0] for pair in ordered_labels[:N]]:
            count += 1
    label_counts.append(count)
print 'TOP %d ACTOR COUNTS' % N
print '{:>20} {:>15}'.format('Actor', '# of Reports')
print '{:>20} {:>15}'.format('-----', '------------')
for label, count in zip(label_index, label_counts):
    print '{:>20} {:>15}'.format(label, count)
with open('TOP_3_ACTOR_COUNTS.txt', 'w') as f:
    f.write('ACTOR, # OF REPORTS\n')
    for label, count in zip(label_index, label_counts):
        f.write(label + ',' + str(count))
