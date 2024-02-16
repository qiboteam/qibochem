"""
A rough Python script attempting to implement the VQE circuit ansatz proposed in Gard et al. using Qibo.

Reference paper: https://doi.org/10.1038/s41534-019-0240-1

Acknowledgements: The original draft of this code, in the form of a Jupyter notebook, was prepared by Caleb Seow from
Eunoia Junior College, who was attached to IHPC in December 2023 under the A*STAR Research Attachment Programme for
Junior College students.
"""

import numpy as np
from qibo import Circuit, gates

from qibochem.driver.molecule import Molecule

# Define molecule
mol = Molecule(xyz_file="h2.xyz")
mol.run_pyscf()


n = 4  # no of qubits
m = 2  # no of electrons
theta = 0.1
phi = 0.1


def special_R(a, theta, phi):
    A = Circuit(n)
    A.add(gates.RY(a, theta + np.pi / 2))
    A.add(gates.RZ(a, phi + np.pi))
    return A


print(special_R(1, 0.1, 0.1).invert().draw())
print(special_R(1, 0.1, 0.1).draw())


def A_gate(a):
    A = Circuit(n)
    A.add(gates.CNOT(a + 1, a))
    A += special_R(a + 1, theta, phi).invert()
    A.add(gates.CNOT(a, a + 1))
    A += special_R(a + 1, theta, phi)
    A.add(gates.CNOT(a + 1, a))
    return A


print(A_gate(0).draw())

Gate2 = Circuit(n)
Gate2.add(gates.X(1))
Gate2.add(gates.X(2))
for i in range(2):
    Gate2 += A_gate(0)
    Gate2 += A_gate(2)
    Gate2 += A_gate(1)

print(Gate2.draw())
