#!/bin/bash

P=(1 2 4 8 10 20)

for i in ${P[*]}; do
    echo "RUNNING: mpiexec -n ${i} python3 primes_mpi4py.py"
    mpiexec -n ${i} python3 primes_mpi4py.py 
done
