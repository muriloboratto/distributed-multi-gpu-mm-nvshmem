#!/bin/sh

for i in 2048 4096 8192 16384 32768
do
    for lib in MMM CCC NNN SSS
    do
        echo "Matrix size = $i | Library = $lib"

        mpirun -np 1 ./mmb 0 $i $lib : \
               -np 1 ./mmb 1 $i $lib : \
               -np 1 ./mmb 2 $i $lib : \
               -np 1 ./mmb 3 $i $lib \
               >> result--2048-32768-1node-4GPUs-${lib}.txt
    done
done