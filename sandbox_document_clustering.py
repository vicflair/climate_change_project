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
    actors.append([country for country in countries
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

# View results
command = ['cp', 'topic-term-distributions.csv.gz', 'results.csv.gz']
subprocess.call(command)
command = ['gunzip', 'results.csv.gz']
subprocess.call(command)
with open(output_folder+'/01500/results.csv', 'r') as f:
    pass
with open(output_folder+'/01500/term-index.txt', 'r') as f:
    terms = f.readlines()
    terms = map(str.rstrip, terms)
