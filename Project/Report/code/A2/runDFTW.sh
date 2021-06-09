#!/bin/bash

cc DFTW.c -o DFTW -fopenmp -lm
> "data.txt"
for nthread in 1 2 4 8 12 16 20 24 28 32
do
    export OMP_NUM_THREADS=$nthread
    echo -n "$nthread : "
    for i in {1..10}
    do
        echo -n "$i, "
        srun -n1 ./DFTW >> "data.txt"
    done
    echo done
done
