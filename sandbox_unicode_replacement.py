import csv
import re


def replace_unicode(text):
    replacement = [
        ['K\xc3\x83\xc2\xa4rnstr\xc3\x83\xc2\xb6m', 'Karnstrom'],  # for Ida Karnstrom
        ['Ra\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdl', 'Raul'],  # for Raul Estrada-Oyuela
        ['T\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdpfer', 'Topfer'],  # for Klaus Topfer
        ['Kjell\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdn', 'Kjellen'],  # for Bo Kjellen
        ['C\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdTE', 'COTE'],  # for COTE D'IVOIRE
        ['L\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdnroth', 'Lonroth'],  # for Mans Lonroth
        ['G\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdmez', 'Gomez'],  # for Carlos Gomez
        ['B<F"Times New Roman">\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbd<F255>cher', 'Bucher'],  # for Thomas Bucher
        ['Uma\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbda', 'Umana'], # for Alvaro Umana
        ['Bergu\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbdo', 'Berguno'],  # for Jorge Berguno
        ['Vi\xc3\x83\xc2\xaf\xc3\x82\xc2\xbf\xc3\x82\xc2\xbda', 'Vina'],  # for Antonio La Vina
        ['\xc3\xa2\xc2\x80\xc2\x9c', '\"'],
        ['\xc3\xa2\xc2\x80\xc2\x9d', '\"'],
        ['\xc3\xa2\xc2\x80\xc2\x99', '\''],
        ['\xc3\xa2\xc2\x80\xc2\x93', '-'],
        ['\xc3\xaf\xc2\xbf\xc2\xbd', '\''],
        ['\xc3\x8b\xc2\x9a', ''],
        ['\xc3\x82\xc2\xa0', ''],
        ['\xc3\x83\xc2\x85', 'A'],
        ['\xc3\x83\xc2\xa9', 'e'],
        ['\xc3\x83\xc2\xa0', 'a'],
        ['\xc3\x83\xc2\xa5', 'a'],
        ['\xe2\x82\xac', 'EUR '],  # euro symbol
        ['\xc2\xa3', 'GBP' ], # pound symbol
        ['\xe2\x80\xaf', ' '],
        ['\xe2\x80\xa6', '...'],
        ['\xc2\x8c', 'i']
        ['\xc2\x82', 'e'],
        ['\xc2\x80', ''],
        ['\xc2\x97', ''],
        ['\xc2\x89', 'E'],
        ['\xc3\x89', 'E'],
        ['\xc2\x93', ''],  # seems to be paired with \xc2\x93, maybe quotes?
        ['\xc2\x94', ' '],
        ['\xc2\x91', '\'']
        ['\xc2\x92', '\''],
        ['\xc2\xb4', '\'']
        ['\xc2\xa8', '\"'],
        ['\xc5\x84', 'n'],
        ['\xc3\xb1', 'n'],
        ['\xc3\xa2', 'a'],  # at least 1 report we have "Climate Change" (typo)
        ['\xc2\x9c', ''],
        ['\xc2\x9d', ''],
        ['\xc2\xa6', ''],
        ['\xc3\x83', 'A'],
        ['\xc2\x85', ''],
        ['\xc3\x9c', 'U'],
        ['\xc3\x82', 'A'],  # at least 1 report where used with temperature wrongly
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
        ['\xcc\x81', 'n']
        ['\xc2\xad', ''],  # Used as space, but incorrectly
        ['\xc3\xa4', 'a'],
        ['\xc3\xba', 'u'],
        ['\xc3\xb8', 'o'],
        ['\xc3\xbc', 'u'],
        ['\xc2\x96', ''],
        ['\xc2\xb8', '']  # used only once, incorrectly as comma
        # ['\xc2\xaf', ''],  # high bar symbol
        # ['\xc2\xbf', ''],  # upside-down question mark
        # ['\xc2\xbd', ''],  # 1/2 symbol
    ]
    for r in replacement:
        if r[0] in text:
            text = text.replace(r[0], r[1])
    return text

with open('../enb/ENB_Reports.csv', 'r') as f:
    data = csv.reader(f, delimiter='\t')
    original_reports = [row[7] for row in data]
    reports = [replace_unicode(orep) for orep in original_reports]
all_unicode_chars = []
for report in reports:
    report_unicode = re.findall('[^\x00-\x7f]+', report)
    all_unicode_chars.append(report_unicode)
unicode_list = list(set(reduce(list.__add__, all_unicode_chars)))
examples = dict()
for uc in unicode_list:
    examples[uc] = [report for report in reports if uc in report]
