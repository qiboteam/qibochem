#! /usr/bin/bash
# Generates data for the bar plot comparing different libraries in double precision

: "${nreps:=10}"
: "${nqubits:=10}"
: "${filename_cpu:=libraries_cpu.dat}"
: "${filename_gpu:=libraries_gpu.dat}"


for circuit in qft variational bv supremacy qv
do
  for library in qibo qiskit qulacs projectq hybridq
  do
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu \
                                              --library $library --nreps $nreps --precision double
    echo
  done
  for library in qibo qiskit-gpu qulacs-gpu hybridq-gpu
  do
    CUDA_VISIBLE_DEVICES=0  python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu \
                                              --library $library --nreps $nreps --precision double
    echo
  done
done
