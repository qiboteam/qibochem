#! /usr/bin/bash
# Generates data for the scaling plot (vs nqubits) using all qibo backends

# Command-line parameters
: "${circuit:=qft}"
: "${precision:=double}"
: "${nreps_a:=20}" # for nqubits < 25
: "${nreps_b:=3}"  # for nqubits >= 25
: "${filename_cpu:=qibo_scaling_cpu.dat}"
: "${filename_gpu:=qibo_scaling_gpu.dat}"


# Qibojit backend
for nqubits in {3..24}
do
  CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu \
                                           --library-options backend=qibojit,platform=cupy \
                                           --nreps $nreps_a --precision $precision
  echo
  CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu \
                                           --library-options backend=qibojit,platform=cuquantum \
                                           --nreps $nreps_a --precision $precision
  echo
  CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu \
                                            --library-options backend=qibojit,platform=numba \
                                            --nreps $nreps_a --precision $precision
  echo
done
for nqubits in {25..31}
do
  CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu \
                                           --library-options backend=qibojit,platform=cupy \
                                           --nreps $nreps_b --precision $precision
  echo
  CUDA_VISIBLE_DEVICES=0 python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu \
                                           --library-options backend=qibojit,platform=cuquantum \
                                           --nreps $nreps_b --precision $precision
  echo
  CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu \
                                            --library-options backend=qibojit,platform=numba \
                                            --nreps $nreps_b --precision $precision
  echo
done


# TensorFlow and QiboTF backends
for nqubits in {3..24}
do
  for backend in tensorflow qibotf
  do
    CUDA_VISIBLE_DEVICES=0  python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu \
                                              --library-options backend=$backend --nreps $nreps_a --precision $precision
    echo
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu \
                                              --library-options backend=$backend --nreps $nreps_a --precision $precision
    echo
  done
done
for nqubits in {25..31}
do
  for backend in tensorflow qibotf
  do
    CUDA_VISIBLE_DEVICES=0  python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_gpu \
                                    --library-options backend=$backend --nreps $nreps_b --precision $precision
    echo
    CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu \
                                              --library-options backend=$backend --nreps $nreps_b --precision $precision
    echo
  done
done


# NumPy backend
for nqubits in {3..24}
do
  CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu \
		                  						          --library-options backend=numpy --nreps $nreps_a --precision $precision
  echo
done
for nqubits in {25..28}
do
  CUDA_VISIBLE_DEVICES="" python compare.py --circuit $circuit --nqubits $nqubits --filename $filename_cpu \
		                  						          --library-options backend=numpy --nreps $nreps_b --precision $precision
  echo
done
