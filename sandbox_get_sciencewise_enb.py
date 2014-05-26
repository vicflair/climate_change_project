import csv

enb_files_list = 'allenb.txt'
with open(enb_files_list, 'r') as f:
    files = f.readlines()
enb_files = map(str.rstrip, files)
folder = 'sw_enb/'
texts = []
for i, enb_file in enumerate(enb_files):
    with open(folder + enb_file, 'r') as f:
        text = f.readlines()
        texts.append(text[-1])

sw_enb_csv = 'sw_enb_reports.csv'
with open(sw_enb_csv, 'w') as f:
    for text in texts:
        f.write('0\t'*7 + text)

with open(sw_enb_csv, 'r') as f:
    data = csv.reader(f, delimiter='\t')
    sw_enb = [row[7] for row in data]