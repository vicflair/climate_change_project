import urllib
import pickle
import re


def download_enb_archive():
    # Get list of all English ENB reports to download
    url_enb_archives = 'http://www.iisd.ca/download/asc/'
    enb_reports_en = get_enb_list(url_enb_archives)
    # Get all texts in whole and split into paragraphs
    all_texts, all_paragraphs = process_enb_reports(url_enb_archives,
                                                    enb_reports_en)
    # Save results
    with open('enb_archives_texts', 'w') as f:
        pickle.dump(all_texts, f)
    with open('enb_archives_paragraphs', 'w') as f:
        pickle.dump(all_paragraphs, f)
    # Write results to dummy .csv format for llda_enb.py
    with open('enb_archives_corpus_texts.csv', 'w') as f:
        for text in all_texts:
            f.write('0\t'*7 + text + '\n')
    with open('enb_archives_corpus_paragraphs.csv', 'w') as f:
        for paragraphs in all_paragraphs:
            for para in paragraphs:
                f.write('\t'*7 + para + '\n')


def find_non_ascii_chars(texts):
    # Show unique set of unicode characters
    all_unicode = []
    for text in all_texts:
        unicode = re.findall('[^\x00-\x7f]+', text)
        all_unicode.append(unicode)
    unicode_set = list(set(reduce(list.__add__, all_unicode)))
    # Find occurrences of each unicode character sequence
    examples = dict()
    for uni in unicode_set:
        examples[uni] = []
        for paragraphs in all_paragraphs:
            for para in paragraphs:
                if uni in para:
                    examples[uni].append(para)
    return unicode_set, examples


def get_enb_list(url_enb_archives):
    filehandle = urllib.urlopen(url_enb_archives)
    content = filehandle.read()
    enb_reports_en = re.findall('(?:>)(enb[\d]+e.txt)(?:<)', content)
    return enb_reports_en


def process_enb_reports(url_enb_archives, enb_reports_en):
    # Get and parse each report text and paragraphs
    all_paragraphs = []
    all_texts = []
    for enb_report in enb_reports_en:
        url_report = url_enb_archives + enb_report
        fh = urllib.urlopen(url_report)
        report_text = fh.read()
        report_text_ascii_only = replace_unicode(report_text)
        # Different types of paragraph formatting
        if '\r\n\r\n' in report_text:
            split_text = report_text.split('\r\n\r\n')
            paragraphs = map(lambda s: s.replace('\r\n', ' '), split_text)
            text = report_text_ascii_only.replace('\r\n', ' ')
        elif '\r\n  \r\n  ' in report_text:
            split_text = report_text.split('\r\n  \r\n  ')
            paragraphs = map(lambda s: s.replace('\r\n', ' '), split_text)
            text = report_text_ascii_only.replace('\r\n', ' ')
        elif '\n\n' in report_text:
            split_text = report_text.split('\n\n')
            paragraphs = map(lambda s: s.replace('\n', ' '), split_text)
            text = report_text_ascii_only.replace('\n', ' ')
        # Common paragraph formatting
        paragraphs = map(lambda s: s.replace('\t', ' '), paragraphs)
        all_paragraphs.append(paragraphs)
        all_texts.append(text)
    return all_texts, all_paragraphs


def replace_unicode(text):
    """ Replace non-ASCII unicode characters in ENB corpus using manually
    determined rules. """
    replacement = [
        ['\x85\x94', '.\"'],
        ['\x91\x91', '\"'],  # open quote
        ['\x92\x92', '\"'],  # close quote ??
        ['\x82', 'e'],  # e with acute accent
        ['\x85', ''],   # ??
        ['\x8a', 'S'],  # S with caron
        ['\x91', '\''],  # apostrophe, open single quote
        ['\x92', '\''],  # close single quote
        ['\x93', '\"'],  # occurs in pairs with \x94
        ['\x94', '\"'],  # occurs in pairs with \x93
        ['\x95', ''],  # bullet point
        ['\x96', '-'],  # dash
        ['\x97', ','],  # tough one, appears to be comma
        ['\x9a', 's'],  # s with caron
        ['\xa3', 'GBP '],  # British pound symbol
        ['\xa4', 'n'],  # n with tilde
        ['\xa5', 'YEN '],  # Japanese yen symbol
        ['\xa9', ''],  # ??
        ['\xab', ''],  # ??
        ['\xb0', ' degrees '],  # degree symbol for geographic coordinates
        ['\xba', ' degrees'],  # degree symbol for Celsius
        ['\xc0', 'A'],  # a with grave accent
        ['\xc1', 'A'],  # a with acute accent
        ['\xc7', 'C'],  # C with cedilla
        ['\xc9', 'E'],  # e with acute accent
        ['\xd1', 'N'],  # N with tilde
        ['\xd3', 'O'],  # O with acute accent
        ['\xd4', 'O'],  # O with circumflex
        ['\xdc', 'U'],  # U with diaeresis
        ['\xe0', 'a'],  # a with grave accent
        ['\xe1', 'a'],  # a with acute accent
        ['\xe2', 'a'],  # a with circumflex
        ['\xe3', 'a'],  # a with tilde
        ['\xe4', 'a'],  # a without accent
        ['\xe5', 'a'],  # a with overring
        ['\xe7', 'c'],  # c with cedilla
        ['\xe8', 'e'],  # e with grave accent
        ['\xe9', 'e'],  # e with acute accent
        ['\xea', 'e'],  # e with circumflex
        ['\xed', 'i'],  # i with acute accent
        ['\xf1', 'n'],  # n with tilde
        ['\xf3', 'o'],  # o with acute accent
        ['\xf4', 'o'],  # o with circumflex
        ['\xf6', 'o'],  # o with diaeresis
        ['\xf8', 'o'],  # o without accent
        ['\xfa', 'u'],  # u with acute accent
        ['\xfc', 'u'],  # u with diaeresis
        ['\xfd', 'y'],  # y with acute accent
        ['\xff', ''],  # ??
    ]
    for r in replacement:
        if r[0] in text:
            text = text.replace(r[0], r[1])
    return text


def main():
    download_enb_archive()

if __name__ == '__main__':
    main()