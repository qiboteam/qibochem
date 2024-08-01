"""
A rough Python script attempting to implement the VQE circuit ansatz proposed in Arrazola  et al. using Qibo.

Reference paper: https://doi.org/10.22331/q-2022-06-20-742

Acknowledgements: The original draft of this code, in the form of a Jupyter notebook, was prepared by Conan Tan from
National Junior College, who was attached to IHPC in December 2023 under the A*STAR Research Attachment Programme for
Junior College students.
"""

import numpy as np
from qibo import Circuit, gates, models

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation

# Excitation function


def checkexcitation(elec, orb):
    s_excitations = [(i, j) for i in range(elec) for j in range(elec, orb) if (i + j) % 2 == 0]
    print(s_excitations)

    d_excitations = [
        (0, 1, k, l)
        for k in range(elec, orb)
        for l in range(k + 1, orb)
        if (1 + k + l) % 2 == 0 and ((k % 2) + (l % 2)) == 1
    ]
    print(d_excitations)
    return s_excitations, d_excitations


# H2

# Define molecule and populate
mol = Molecule(xyz_file="h2.xyz")
mol.run_pyscf()
n_qubits = mol.nso
n_electrons = 2
hamiltonian = mol.hamiltonian()
s_excitations, d_excitations = checkexcitation(n_electrons, n_qubits)


## Circuit construction
c = hf_circuit(n_qubits, n_electrons)


def build(c, singleex, doubleex, x):
    for qa, qb in singleex:
        sc = Circuit(n_qubits)
        sc.add(gates.CNOT(qa, qb))
        sc.add(gates.RY(qa, theta=x / 2))
        sc.add(gates.CNOT(qb, qa))
        sc.add(gates.RY(qa, theta=-x / 2))
        sc.add(gates.CNOT(qb, qa))
        sc.add(gates.CNOT(qa, qb))
        c += sc
        # for _i, parameter in enumerate(c.get_parameters()):
        #     gradient = parameter_shift(c, hamiltonian, parameter_index=_i)
        #     print(f"Excitation {qa, qb} => Gradient: {gradient}")
        #     if np.abs(gradient) > 1e-10:
        #         gradients[(qa, qb)] = gradient
        #     break
    for qa, qb, qc, qd in doubleex:
        dc = Circuit(n_qubits)
        dc.add(gates.CNOT(qc, qd))
        dc.add(gates.CNOT(qa, qc))
        dc.add(gates.H(qa))
        dc.add(gates.H(qd))
        dc.add(gates.CNOT(qa, qb))
        dc.add(gates.CNOT(qc, qd))
        dc.add(gates.RY(qa, theta=-x / 8))
        dc.add(gates.RY(qb, theta=x / 8))
        dc.add(gates.CNOT(qa, qd))
        dc.add(gates.H(qd))
        dc.add(gates.CNOT(qd, qb))
        dc.add(gates.RY(qa, theta=-x / 8))
        dc.add(gates.RY(qb, theta=x / 8))
        dc.add(gates.CNOT(qc, qb))
        dc.add(gates.CNOT(qc, qa))
        dc.add(gates.RY(qa, theta=x / 8))
        dc.add(gates.RY(qb, theta=-x / 8))
        dc.add(gates.CNOT(qd, qb))
        dc.add(gates.H(qd))
        dc.add(gates.CNOT(qa, qd))
        dc.add(gates.RY(qa, theta=x / 8))
        dc.add(gates.RY(qb, theta=-x / 8))
        dc.add(gates.CNOT(qa, qb))
        dc.add(gates.CNOT(qc, qa))
        dc.add(gates.H(qa))
        dc.add(gates.H(qd))
        dc.add(gates.CNOT(qa, qc))
        dc.add(gates.CNOT(qc, qd))
        c += dc
        # for _i, parameter in enumerate(c.get_parameters()):
        #     gradient = parameter_shift(c, hamiltonian, parameter_index=_i)
        #     print(f"Excitation {qa, qb, qc, qd} => Gradient: {gradient}")
        #     if np.abs(gradient) > 1e-10:
        #         gradients[(qa, qb, qc, qd)] = gradient
        #     break
    return c


print(list(enumerate(c.get_parameters())))
c = build(c, s_excitations, d_excitations, 0.1)
print(c.draw())
print(c.summary())
