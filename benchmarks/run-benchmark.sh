#!/bin/bash

#
# Test variables
#
data_folder="data"
nreps=5
precisions="double" # "single double"

# backends="numpy qibojit tensorflow pytorch"
backends="tensorflow" # ZC note: pytorch backend not fully implemented

# platforms="numba cupy cuquantum"
platforms="cupy cuquantum" # ZC note: If using numba platform with GPU available, will it use the GPU or stick to CPU?

n_hydrogens="4" # "2 4 6 8 10"

# echo "Main script starts"
# Run main.py with various command-line arguments/options
# for n_hydrogen in 2 4 6 8; do
for n_hydrogen in ${n_hydrogens}; do
    # echo ${n_hydrogen}
    for backend in ${backends}; do
    # for backend in numpy qibojit; do
        if [ "${backend}" == "qibojit" ]; then
            for platform in ${platforms}; do
            # for platform in numba cupy cuquantum; do
                filename="h${n_hydrogen}_${backend}_${platform}"
                for precision in ${precisions}; do
                # for precision in single double; do
                    python main.py --n_hydrogens ${n_hydrogen} --backend ${backend} --platform ${platform} --nreps ${nreps} --precision ${precision} --filename ${filename}.dat &>> ${filename}.log
                    echo &>> ${filename}.log
                done

                # Move the .log and .dat file into data/ folder
                if [ -d "${data_folder}" ]; then
                    mv ${filename}.dat ${filename}.log "${data_folder}"
                fi
            done
        else
            filename="h${n_hydrogen}_${backend}"
            for precision in ${precisions}; do
                python main.py --n_hydrogens ${n_hydrogen} --backend ${backend} --nreps ${nreps} --precision ${precision} --filename ${filename}.dat &>> ${filename}.log
                echo &>> ${filename}.log
            done

            # Move the .log and .dat file into data/ folder
            if [ -d "data" ]; then
                mv ${filename}.dat ${filename}.log "${data_folder}"
            fi
        fi
    done
done
