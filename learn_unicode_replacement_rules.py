# Script to help manually explore and curate replacement rules for the
# non-ASCII characters found in the ENB corpus. The strategy is to search all
# occurrences of non-ASCII characters and guess replacements. Rules which
# involve longer sequences of non-ASCII characters are given priority, as it
# seems that these long sequences themselves constitute their own substitution
# different from any sub-sequences contained in them, e.g. some names with
# accents are messed up.

import csv
import re


def replace_unicode(text):
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

def analyze_unicode():
    """ Analyze ENB corpus using current manual replacement rules for
     unicode characters and return analysis data."""
    # Determine reports using current manual rules for comparison and to
    # filter out non-ASCII characters which have already been "solved"
    with open('../enb/ENB_Reports.csv', 'r') as f:
        data = csv.reader(f, delimiter='\t')
        original_reports = [row[7] for row in data]
        reports = [replace_unicode(orep) for orep in original_reports]
    # Get the list of all unique non-ASCII unicode character sequences
    all_unicode_chars = []
    for report in reports:
        report_unicode = re.findall('[^\x00-\x7f]+', report)
        all_unicode_chars.append(report_unicode)
    unicode_list = list(set(reduce(list.__add__, all_unicode_chars)))
    # Get a list of all reports containing each of the unique unicode sequences
    examples = dict()
    for uc in unicode_list:
        examples[uc] = [report for report in reports if uc in report]
    # Return results for analysis
    return original_reports, reports, unicode_list, examples