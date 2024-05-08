#! /usr/bin/bash
# Generate logs for library comparison on different circuits with gate fusion enabled

# Command-line parameters
: "${filename_cpu:=libraries_fusion_cpu.dat}"
: "${filename_gpu:=libraries_fusion_gpu.dat}"
: "${precision:=single}"
: "${nqubits:=30}"
: "${nreps_cpu:=3}"
: "${nreps_gpu:=5}"


for circuit in qft variational supremacy bv qv
do
    # GPU benchmarks
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu --library qibo \
                                             --library-options backend=qibojit,platform=cupy,max_qubits=2 --nreps $nreps_gpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu --library qibo \
                                             --library-options backend=qibojit,platform=cuquantum,max_qubits=2 --nreps $nreps_gpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu --library qsim-gpu \
                                             --library-options max_qubits=2 --nreps $nreps_gpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu --library qsim-cuquantum \
                                             --library-options max_qubits=2 --nreps $nreps_gpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu --library qiskit-gpu \
                                             --library-options max_qubits=2 --nreps $nreps_gpu --precision $precision
    echo

    # CPU benchmarks
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu --library qibo \
                                              --library-options backend=qibojit,platform=numba,max_qubits=2 --nreps $nreps_cpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu --library qsim \
                                              --library-options max_qubits=2 --nreps $nreps_cpu --precision $precision
    echo
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu --library qiskit \
                                              --library-options max_qubits=2 --nreps $nreps_cpu --precision $precision
    echo
done
