"""
Parse the .dat files from running main.py to get and plot the dry_run_times
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_dat_files(dat_glob, directory=None):
    """
    Read all the .dat files found with dat_glob. .dat files should be in .json format
    """
    if directory is None:
        directory = "."

    result = {}
    for dat_file in Path(directory).glob(dat_glob):
        _h_n, backend = dat_file.stem.split("_")
        h_n = int(_h_n[1:])
        if result.get(h_n) is None:
            result[h_n] = {}
        with open(dat_file) as file_handler:
            result[h_n][backend] = json.load(file_handler)
    return result


def main():
    dat_glob = "*.dat"
    json_data = parse_dat_files(dat_glob)
    # Plot results
    x_vals = sorted(json_data.keys())
    # print(x_vals)
    qibojit_vals = [json_data[_i]["qibojit"][0]["dry_run_time"] for _i in x_vals]
    # print(qibojit_vals)
    numpy_vals = [json_data[_i]["numpy"][0]["dry_run_time"] for _i in x_vals]
    # print(numpy_vals)

    plt.plot(x_vals, qibojit_vals, label="Qibojit")
    plt.plot(x_vals, numpy_vals, label="Numpy")
    plt.legend()
    plt.ylabel("Time (s)")
    plt.xlabel(r"Length of hydrogen chain, $H_n$")
    plt.savefig("benchmark.svg")


if __name__ == "__main__":
    main()
