"""
A rough Python script attempting to implement the VQE circuit ansatz proposed in Arrazola  et al. using Qibo.

Reference paper: https://doi.org/10.22331/q-2022-06-20-742

Acknowledgements: The original draft of this code, in the form of a Jupyter notebook, was prepared by Caleb Seow from
Eunoia Junior College, who was attached to IHPC in December 2023 under the A*STAR Research Attachment Programme for
Junior College students.
"""

import numpy as np
from qibo import Circuit, gates, models

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation

mol = Molecule(xyz_file="h2.xyz")
mol.run_pyscf()
n = 4

# Single Excitation

g1 = hf_circuit(mol.nso, 2)

theta = 0.1


def single_ex(a, b):
    g = Circuit(4)
    g.add(gates.CNOT(a, b))
    g.add(gates.RY(a, theta))
    g.add(gates.CNOT(b, a))
    g.add(gates.RY(a, -theta))
    g.add(gates.CNOT(b, a))
    g.add(gates.CNOT(a, b))
    return g


for i in range(2):
    for j in range(2, 4):
        if (i + j) % 2 == 0:
            g1 += single_ex(i, j)

print(g1.summary())
print(g1.draw())

theta = 0.1


def double_ex():
    g = Circuit(4)
    g.add(gates.CNOT(2, 3))
    g.add(gates.CNOT(0, 2))
    g.add(gates.H(0))
    g.add(gates.H(3))
    g.add(gates.CNOT(0, 1))  # 2 simultaneous CNOTs?
    g.add(gates.CNOT(2, 3))
    g.add(gates.RY(0, -theta))
    g.add(gates.RY(1, theta))
    g.add(gates.CNOT(0, 3))
    g.add(gates.H(3))
    g.add(gates.CNOT(3, 1))
    g.add(gates.RY(0, -theta))
    g.add(gates.RY(1, theta))
    g.add(gates.CNOT(2, 1))
    g.add(gates.CNOT(2, 0))
    g.add(gates.RY(0, theta))
    g.add(gates.RY(1, -theta))
    g.add(gates.CNOT(3, 1))
    g.add(gates.H(3))
    g.add(gates.CNOT(0, 3))
    g.add(gates.RY(0, theta))
    g.add(gates.RY(1, -theta))
    g.add(gates.CNOT(0, 1))
    g.add(gates.CNOT(2, 0))
    g.add(gates.H(0))
    g.add(gates.H(3))
    g.add(gates.CNOT(0, 2))
    g.add(gates.CNOT(2, 3))
    return g


g1 += double_ex()

print(g1.draw())

hamiltonian = mol.hamiltonian()
# parameter = float(input("theta = "))
parameter = 0.1
params = np.zeros(len(g1.get_parameters())) + parameter
g1.set_parameters(params)

energy = expectation(g1, hamiltonian)
print(f"Expectation: {energy}")

vqe = models.VQE(g1, hamiltonian)

best, params, extra = vqe.minimize(params, method="BFGS", compile=False)
# Methods: BFGS, COBYLA, Powell

print(f"Energy: {best}")
exact_result = hamiltonian.eigenvalues()[0]
print(f"Exact result: {exact_result}")

hamiltonian = mol.hamiltonian()
# parameter = float(input("theta = "))
parameter = 0.1
params = np.zeros(len(g1.get_parameters())) + parameter
g1.set_parameters(params)

energy = expectation(g1, hamiltonian)
print(f"Expectation: {energy}")

vqe = models.VQE(g1, hamiltonian)

best, params, extra = vqe.minimize(params, method="BFGS", compile=False)
# Methods: BFGS, COBYLA, Powell

print(f"Energy: {best}")

print(f"Exact result: {exact_result}")
