#! /usr/bin/bash
# Generates data for the evolution scaling with dt using the dense form of the Hamiltonian

# Command-line parameters
: "${nqubits:=10}"
: "${precision:=double}"
: "${filename:=evolution.dat}"
: "${nreps:=5}"


# GPU backends
for dt in 0.1 0.095 0.09 0.085 0.08 0.075 0.07 0.065 0.06 0.055 0.05 0.045 \
          0.04 0.035 0.03 0.025 0.02 0.015 0.01 0.005
do
  for backend in qibojit tensorflow
  do
    CUDA_VISIBLE_DEVICES=0 python evolution.py --nqubits $nqubits --dt $dt --filename $filename --nreps $nreps \
  		                  	  					         --backend $backend --precision $precision --dense
    echo
  done
done

# CPU backends
for dt in 0.1 0.095 0.09 0.085 0.08 0.075 0.07 0.065 0.06 0.055 0.05 0.045 \
          0.04 0.035 0.03 0.025 0.02 0.015 0.01 0.005
do
  for backend in numpy tensorflow
  do
    CUDA_VISIBLE_DEVICES="" python evolution.py --nqubits $nqubits --dt $dt --filename $filename --nreps $nreps \
  		                  						            --backend $backend --precision $precision --dense
    echo
  done
done
