#! /usr/bin/bash
# Generates data for qibojit breakdown bar plot with dry run vs simulation

# Command-line parameters
: "${filename:=qibojit_breakdown.dat}"
: "${precision:=double}"
: "${circuit:=supremacy}"
: "${nreps_cpu:=5}"
: "${nreps_gpu:=10}"


for nqubits in 16 18 20 22 24 26 28
do
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                             --library-options backend=qibojit,platform=cupy --nreps $nreps_gpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                             --library-options backend=qibojit,platform=cuquantum --nreps $nreps_gpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                              --library-options backend=qibojit,platform=numba --nreps $nreps_cpu --precision $precision
    echo
done
