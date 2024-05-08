"""Launches the circuit benchmark script for user given arguments."""

import argparse

from benchmarks.scripts import circuit_benchmark

THREADING_LINK = "https://numba.pydata.org/numba-doc/latest/user/threading-layer.html#selecting-a-named-threading-layer"

parser = argparse.ArgumentParser()
parser.add_argument("--n_hydrogens", default=2, type=int, help="Number of hydrogen atoms in the system")
# parser.add_argument("--nqubits", default=20, type=int,
#                     help="Number of qubits in the circuit.")
parser.add_argument("--backend", default="qibojit", type=str, help="Qibo backend to use for simulation.")
parser.add_argument("--platform", default=None, type=str, help="Qibo platform to use for simulation.")

# parser.add_argument("--circuit", default="qft", type=str,
#                     help="Type of circuit to use. See README for the list of "
#                          "available circuits.")
# parser.add_argument("--circuit-options", default=None, type=str,
#                     help="String with options for circuit creation. "
#                          "It should have the form 'arg1=value1,arg2=value2,...'. "
#                          "See README for the list of arguments that are "
#                          "available for each circuit.")

parser.add_argument(
    "--nreps", default=1, type=int, help="Number of repetitions of the circuit execution. " "Dry run is not included."
)
parser.add_argument(
    "--nshots",
    default=None,
    type=int,
    help="Number of measurement shots. If used the time "
    "required to measure frequencies (no samples) is "
    "measured and logged. If it is ``None`` no "
    "measurements are performed.",
)
parser.add_argument(
    "--transfer",
    action="store_true",
    help="If used the final state array is converted to numpy. "
    "If the simulation device is GPU this requires a "
    "transfer from GPU memory to CPU.",
)

parser.add_argument(
    "--precision",
    default="double",
    type=str,
    help="Numerical precision of the simulation. " "Choose between 'double' and 'single'.",
)
parser.add_argument(
    "--memory",
    default=None,
    type=int,
    help="Limit the GPU memory usage when using Tensorflow "
    "based backends. The memory limit should be given "
    "in MB. Tensorflow reserves the full available memory "
    "by default.",
)
parser.add_argument(
    "--threading",
    default=None,
    type=str,
    help="Switches the numba threading layer when using the "
    "qibojit backend on CPU. See {} for a list of "
    "available threading layers.".format(THREADING_LINK),
)

parser.add_argument(
    "--filename",
    default=None,
    type=str,
    help="Directory of file to save the logs in json format. "
    "If not given the logs will only be printed and not saved.",
)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    # args["circuit_name"] = args.pop("circuit")
    circuit_benchmark(**args)
