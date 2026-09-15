#!/bin/sh

for i in 2048 4096 8192 16384 32768
do
  mpirun -np 1 ./mmb 0 $i MMM : -np 1 ./mmb 1 $i MMM : -np 1 ./mmb 2 $i MMM : -np 1 ./mmb 3 $i MMM >> result--2048-32768-1node-4GPUs-MMM.txt
  mpirun -np 1 ./mmb 0 $i CCC : -np 1 ./mmb 1 $i CCC : -np 1 ./mmb 2 $i CCC : -np 1 ./mmb 3 $i CCC >> result--2048-32768-1node-4GPUs-CCC.txt
  mpirun -np 1 ./mmb 0 $i NNN : -np 1 ./mmb 1 $i NNN : -np 1 ./mmb 2 $i NNN : -np 1 ./mmb 3 $i NNN >> result--2048-32768-1node-4GPUs-NNN.txt
  mpirun -np 1 ./mmb 0 $i SSS : -np 1 ./mmb 1 $i SSS : -np 1 ./mmb 2 $i SSS : -np 1 ./mmb 3 $i SSS >> result--2048-32768-1node-4GPUs-SSS.txt
done

