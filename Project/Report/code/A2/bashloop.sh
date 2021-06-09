#!/bin/bash

for nthread in 1 2 4 8 12 16 20 24 28 32
do
    export OMP_NUM_THREADS=$nthread
    > "data_${OMP_NUM_THREADS}.txt"
    for i in {1..5}
    do
        cc -O2 -fopenmp stream.c -o stream_omp
        srun -n 1 ./stream_omp >> "data_${OMP_NUM_THREADS}.txt"
    done
done