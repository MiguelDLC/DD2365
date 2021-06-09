#!/bin/bash
cc -O3 -fopenmp bonus.c -o bonus
> "data.txt"
for nthread in {1..32..1}
do
    export OMP_NUM_THREADS=$nthread
    echo "Nthreads = $nthread"
    for i in {1..10}
    do
        srun -n1 ./swe >> "data.txt"
    done
done
