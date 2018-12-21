import math
import time
# here is MPI.Init()
from mpi4py import MPI

from itertools import accumulate, chain

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# we are looking for prime numbers in this interval
lower_bound = 1000000000000
upper_bound = 1000000000100

# tested function returning number of factors of n
def number_of_factors(n):
    j = 0
    for i in range(2, math.floor(math.sqrt(n))):
        if n % i == 0:
            j += 1
    return j


def check_numbers_in_interval(lower, upper):
    out = []
    for number in range(lower, upper):
        if number_of_factors(number) == 0:
            out.append(number)
    return out
    

def prepare_intervals(lower, upper, p):
    samples = upper - lower
    integer = math.floor(samples / p)
    rem = samples % p
    upper_bounds = list(accumulate([integer + 1 if rem > i else integer for i in range(p)]))
    lower_bounds = list([0] + [x + 1 for x in upper_bounds[:-1]])
    # a list [(0, i), (i+1, 2*i), ..] is returned
    return list(zip(lower_bounds, upper_bounds))


if __name__ == '__main__':
    intervals = None

    if rank == 0:
        start = time.perf_counter()
        intervals = prepare_intervals(lower_bound, upper_bound, size)
    
    # the data is a tuple (lower, upper)
    lower, upper = comm.scatter(intervals, root=0)
    data = check_numbers_in_interval(lower_bound + lower, lower_bound + upper)
    # print('process: ', rank, 'data: ', (lower, upper))
    out = comm.gather(data, root=0)

    if rank == 0:
        end = time.perf_counter()
        print(list(chain(*out)))
        print('time: ', end - start)
        # MPI.Finalize() automatically at exit
