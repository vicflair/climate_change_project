from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import urllib
import re
import time


# I/O bound task: Fetching files from the internet
def get_enb_list(url_enb_archives='http://www.iisd.ca/download/asc/'):
    filehandle = urllib.urlopen(url_enb_archives)
    content = filehandle.read()
    enb_reports_en = re.findall('(?:>)(enb[\d]+e.txt)(?:<)', content)
    reports = map(lambda s: url_enb_archives + s, enb_reports_en)
    return reports


# CPU bound task: factorizing to prime numbers
def factorize_naive(n):
    """ A naive factorization method. Take integer 'n', return list of
        factors.
    """
    if n < 2:
        return []
    factors = []
    p = 2
    while True:
        if n == 1:
            return factors
        r = n % p
        if r == 0:
            factors.append(p)
            n = n / p
        elif p * p >= n:
            factors.append(n)
            return factors
        elif p > 2:
            # Advance in steps of 2 over odd numbers
            p += 2
        else:
            # If p == 2, get to 3
            p += 1
    assert False, "unreachable"


def serial_factorizer(nums):
    start = time.time()
    for n in nums:
        factorize_naive(n)
    end = time.time()
    print '1 process: {0} secs'.format(end - start)


def parallel_factorizer(nums, num_threads):
    start = time.time()
    p = Pool(num_threads)
    results = p.map(factorize_naive, nums)
    p.close()
    p.join()
    end = time.time()
    print '{0} processes: {1} secs'.format(num_threads, end - start)


def compare_num_processes():
    n = range(1000000)
    serial_factorizer(n)
    parallel_factorizer(n, 2)
    parallel_factorizer(n, 4)
    parallel_factorizer(n, 8)
    parallel_factorizer(n, 12)
    parallel_factorizer(n, 16)
    parallel_factorizer(n, 32)


def time_io_p(reports, num_threads):
    """ Multiple processes
    """
    start = time.time()
    p = Pool(num_threads)
    results = p.map(urllib.urlopen, reports)
    p.close()
    p.join()
    end = time.time()
    print '{0} threads: {1} secs'.format(n, end - start)


def time_io_t(reports, n):
    """ Multiple threads
    """
    start = time.time()
    t = ThreadPool(n)
    results = t.map(urllib.urlopen, reports)
    t.close()
    t.join()
    end = time.time()
    print end - start


def time_io_s(reports):
    """ Single process, single thread
    """
    start = time.time()
    results = []
    for report in reports:
        results.append(urllib.urlopen(report))
    end = time.time()
    print end - start


def compare_num_threads(reports):
    times(reports)
    timet(reports, 4)
    timet(reports, 8)
    timet(reports, 12)


if __name__ == "__main__":
    pass