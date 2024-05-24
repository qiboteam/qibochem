"""
Parse the .dat files from running main.py to get and plot the simulation_times_mean
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_dat_files(dat_glob, directory):
    """
    Read all the .dat files found with dat_glob. .dat files should be in .json format
    """
    result = {}
    for dat_file in Path(directory).glob(dat_glob):
        _h_n, *_backend = dat_file.stem.split("_")
        # Combine backend str to match the .json keys
        # print(_backend)
        backend = _backend[0] if len(_backend) == 1 else f"{_backend[0]} ({_backend[1]})"
        # print(backend)
        h_n = int(_h_n[1:])
        if result.get(h_n) is None:
            result[h_n] = {}
        with open(dat_file) as file_handler:
            result[h_n][backend] = json.load(file_handler)
    return result


def main(plot_result, print_result, directory=None):
    if directory is None:
        directory = "."

    dat_glob = "*.dat"
    json_data = parse_dat_files(dat_glob, directory)
    # for _k, _v in json_data.items():
    #     print(_k)
    #     print(list(_v.keys()))
    #     print()
    # return

    # Global list of backends that intending to test
    backends = ["numpy", "qibojit", "qibojit (numba)", "tensorflow", "qibojit (cupy)", "qibojit (cuquantum)"]
    sim_time = "simulation_times_mean"

    # Plot results
    all_x_vals = sorted(json_data.keys())
    # print(all_x_vals)
    # print()
    # print(json_data[4])
    # print()
    # print(json_data[4]["qibojit"])

    all_results = {backend: {} for backend in backends}
    for backend in backends:
        x_vals = [_i for _i in all_x_vals if json_data[_i].get(backend) is not None]
        # for _i in x_vals:
        #     print(_i, backend)
        #     print(json_data[_i][backend][0])
        #     print()

        y_vals = [json_data[_i][backend][0][sim_time] for _i in x_vals]
        all_results[backend]["x"] = x_vals
        all_results[backend]["y"] = y_vals

    # print(all_results)
    # return

    # TODO: Finish making nice the printed output!
    if print_result:
        print("Timings:")
        print("________________________________")
        backend_str = " | ".join(backends)
        print(f"| nQubits | {backend_str} |")
        print("|------------------------------|")
        # for n_h, np_result, jit_result in zip(x_vals, numpy_vals, qibojit_vals):
        #     print(f"|{n_h: ^10}| {np_result:7.3f} | {jit_result:7.3f} |")
        print("|______________________________|\n")

    if plot_result:
        for backend, data in all_results.items():
            plt.scatter(data["x"], data["y"], label=backend)
        plt.legend()
        plt.ylabel("Time (s)")
        plt.xlabel(r"Length of hydrogen chain, $H_n$")
        plt.savefig("benchmark.svg")


if __name__ == "__main__":
    directory = None  # "test1"
    main(plot_result=True, print_result=False, directory=directory)
