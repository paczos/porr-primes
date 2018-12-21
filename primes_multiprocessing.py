import multiprocessing
import math
import time

from multiprocessing import Queue
from itertools import accumulate

# we are looking for prime numbers in this interval
lower_bound = 1739834324567
upper_bound = 1739834324667 

P = [1, 2, 3, 4, 8, 10, 20]

# for communication purposes, we need a thread-safe data structure
q = Queue()


# tested function returning number of factors of n
def number_of_factors(n):
    j = 0
    for i in range(2, math.floor(math.sqrt(n))):
        if n % i == 0:
            j += 1
    return j


def check_numbers_in_interval(lower, upper):
    for number in range(lower, upper):
        if number_of_factors(number) == 0:
            q.put(number)
    

def prepare_intervals(lower, upper, p):
    samples = upper - lower
    integer = math.floor(samples / p)
    rem = samples % p
    return list(accumulate([0] + [integer + 1 if rem > i else integer for i in range(p)]))


def main(intervals, p):
    processes = [ ]
    for i in range(p):
        p = multiprocessing.Process(target=check_numbers_in_interval, args=(lower_bound + intervals[i], lower_bound + intervals[i+1]))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()


if __name__ == '__main__':
    for processes in P:
        intervals = prepare_intervals(lower_bound, upper_bound, processes)
        print('processes: ', processes, 'intervals: ', intervals)
        start = time.perf_counter()
        main(intervals, processes)
        end = time.perf_counter()
        print('time: ', end - start)

        primes = [ ]
        while not q.empty():
            primes.append(q.get())

        print(primes)
