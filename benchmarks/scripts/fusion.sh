#! /usr/bin/bash
# Generate logs for fusion comparison on different circuits for all qibojit platforms

# Command-line parameters
: "${filename:=qibojit_fusion.dat}"
: "${precision:=double}"
: "${nqubits:=30}"
: "${nreps_cpu:=3}"
: "${nreps_gpu:=5}"


for circuit in qft variational supremacy bv qv
do
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                             --library-options backend=qibojit,platform=cupy,max_qubits=2 --nreps $nreps_gpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                             --library-options backend=qibojit,platform=cupy --nreps $nreps_gpu --precision $precision
    echo

    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                             --library-options backend=qibojit,platform=cuquantum,max_qubits=2 --nreps $nreps_gpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                             --library-options backend=qibojit,platform=cuquantum --nreps $nreps_gpu --precision $precision
    echo

    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                              --library-options backend=qibojit,platform=numba,max_qubits=2 --nreps $nreps_cpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename \
                                              --library-options backend=qibojit,platform=numba --nreps $nreps_cpu --precision $precision
    echo
done
