#! /usr/bin/bash
# Generates data for the bar plot comparing different libraries in single precision

: "${nreps:=10}"
: "${nqubits:=20}"
: "${filename_cpu:=libraries_cpu.dat}"
: "${filename_gpu:=libraries_gpu.dat}"


for circuit in qft variational bv supremacy qv
do
  for library in qibo qiskit qsim hybridq
  do
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu \
                                              --library $library --nreps $nreps --precision single
    echo
  done
  for library in qibo qiskit-gpu qsim-gpu qsim-cuquantum qcgpu hybridq-gpu
  do
     CUDA_VISIBLE_DEVICES=0  python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu \
                                               --library $library --nreps $nreps --precision single
    echo
	done
done
