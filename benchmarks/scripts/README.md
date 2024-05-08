# Example scripts

This folder contains example bash scripts that execute the `compare.py` benchmark for different circuit configurations and different libraries.
The available circuits are described in the main ``README.md`` of this repository
(some of them are presented in Table 1 of the [HyQuas paper](https://dl.acm.org/doi/pdf/10.1145/3447818.3460357)).

The provided scripts are the following:

### `qibo.sh`

Execute a specific circuit with different sizes and different Qibo backends.
By default, it executes the circuit with all backends (NumPy, TensorFlow, Qibotf and Qibojit) and with circuit sizes from 3 to 31 (28 for NumPy).
If different platforms are available, it will use the default one.
It is straightforward to edit the bash script to fine tune the variation ranges and the number of repetitions for each circuit size.
A starter code to plot the results is available in ``plots/qibo.ipynb``.
Options:
 - ``circuit``: circuit to execute (default: ``qft``)
 - ``precision``: ``single`` or ``double`` (default: ``double``)

### ``qibojit.sh``

Similar to ``qibo.sh``, but uses only qibojit. Useful to compare the performance across different platforms and different GPUs.
A starter code to plot the results is available in ``plots/qibojit.ipynb``.
Options:
 - ``filename``: where to store the logs (default: ``qibojit.dat``)
 - ``circuit``: circuit to execute (default: ``qft``)
 - ``precision``: ``single`` or ``double`` (default: ``double``)
 - ``nreps``: number of repetitions for each circuit size (default: ``10``)
 - ``platform``: platform to use for simulation (default: ``cupy``)

### ``qibo_circuits.sh``

Execute a set of circuits for different circuit sizes, using a specific backend.
Useful to plot the breakdown of execution times (starter code in ``plots/qibo.ipynb``).
Options:
 - ``backend``: backend to use for simulation (default: ``qibojit``)
 - ``precision``: ``single`` or ``double`` (default: ``double``)
 - ``nreps``: number of repetitions for each circuit and size (default: ``10``)

### ``library_single.sh``

Execute a set of circuits for different circuit sizes, using a variety of different libraries.
This script focuses on libraries that support single precision simulation.
It is straightforward to edit the bash script to change the circuit sizes and the libraries.
A starter code to plot the results is available in ``plots/libraries.ipynb``.
Options:
 - ``nreps``: number of repetitions for each circuit size (default: ``10``)

### ``library_double.sh``

Same as ``library_single.sh``, but with libraries that support double precision simulation.
Options:
 - ``nreps``: number of repetitions for each circuit size (default: ``10``)
