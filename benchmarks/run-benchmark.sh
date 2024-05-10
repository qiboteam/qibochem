#!/bin/bash


nreps=10

# Run main.py with various command-line arguments/options
for n_hydrogen in 2 4 6 8; do
    for backend in numpy qibojit; do
        filename="h${n_hydrogen}_${backend}"
        for precision in single double; do
            python main.py --n_hydrogens ${n_hydrogen} --backend ${backend} --nreps ${nreps} --precision ${precision} --filename ${filename}.dat &>> ${filename}.log
            echo &>> ${filename}.log
        done

        # Move the .log and .dat file into data/ folder
        if [ -d "data" ]; then
            mv ${filename}.dat ${filename}.log
        fi
    done
done
