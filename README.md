# Qibochem

![Tests](https://github.com/qiboteam/qibochem/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibochem/graph/badge.svg?token=2CMDZP1GU2)](https://codecov.io/gh/qiboteam/qibochem)

Qibochem is a plugin to [Qibo](https://github.com/qiboteam/qibo) for quantum chemistry simulations.

Qibochem key features

* General purpose Molecule class
  * PySCF or Psi4 for calculation of 1- and 2-electron integrals
  * User defined orbital active space
* Unitary Coupled Cluster Ansatz
* Various Qibo backends (numpy, JIT, TN) for efficient simulation

## Installation

Using poetry

```
git clone https://github.com/qiboteam/qibochem.git
cd qibochem
poetry install
```

## Contributing

Contributions, issues and feature requests are welcome.
