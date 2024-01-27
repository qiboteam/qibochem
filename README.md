# Qibochem

![Tests](https://github.com/qiboteam/qibochem/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibochem/graph/badge.svg?token=2CMDZP1GU2)](https://codecov.io/gh/qiboteam/qibochem)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10473173.svg)](https://doi.org/10.5281/zenodo.10473173)

Qibochem is a plugin to [Qibo](https://github.com/qiboteam/qibo) for quantum chemistry simulations.

Some of the features of Qibochem are:

* General purpose `Molecule` class
  * PySCF for calculating the molecular  1- and 2-electron integrals
  * User defined orbital active space
* Unitary Coupled Cluster Ansatz
* Various Qibo backends (numpy, JIT, TN) for efficient simulation

## Documentation

The Qibochem documentation can be found [here](https://qibo.science/qibochem/stable)

## Minimum working example:

An example of building the UCCD ansatz with a H2 molecule

```
import numpy as np
from qibo.models import VQE

from qibochem.driver.molecule import Molecule
from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.ucc import ucc_circuit

# Define the H2 molecule and obtain its 1-/2- electron integrals with PySCF
h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))])
h2.run_pyscf()
# Generate the molecular Hamiltonian
hamiltonian = h2.hamiltonian()

# Build a UCC circuit ansatz for running VQE
circuit = hf_circuit(h2.nso, sum(h2.nelec))
circuit += ucc_circuit(h2.nso, [0, 1, 2, 3])

# Create and run the VQE, starting with random initial parameters
vqe = VQE(circuit, hamiltonian)

initial_parameters = np.random.uniform(0.0, 2*np.pi, 8)
best, params, extra = vqe.minimize(initial_parameters)
```

## Citation policy

If you use the Qibochem plugin please refer to the documentation for citation instructions.

## Contact

For questions, comments and suggestions please contact us at [https://matrix.to/#/#qibo:matrix.org](url)

## Contributing

Contributions, issues and feature requests are welcome.
