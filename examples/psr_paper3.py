"""
This script is a continuation of 'paper3_conan.py'. After implementing the circuit ansatz, this script attempts to
examine the use of the Parameter Shift Rule implemented in Qibo for chemistry applications. In addition, simulations of
LiH are used for some benchmarking.

Reference paper: https://doi.org/10.22331/q-2022-06-20-742

Acknowledgements: The original draft of this code, in the form of a Jupyter notebook, was prepared by Conan Tan from
National Junior College, who was attached to IHPC in December 2023 under the A*STAR Research Attachment Programme for
Junior College students.
"""

import numpy as np
from qibo import Circuit, gates
from qibo.derivative import parameter_shift
from qibo.optimizers import optimize

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
n_electrons = mol.nelec
hamiltonian = mol.hamiltonian()
s_excitations, d_excitations = checkexcitation(n_electrons, n_qubits)
gradients = {}

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
        for _i, parameter in enumerate(c.get_parameters()):
            gradient = parameter_shift(c, hamiltonian, parameter_index=_i)
            print(f"Excitation {qa, qb} => Gradient: {gradient}")
            if np.abs(gradient) > 1e-10:
                gradients[(qa, qb)] = gradient
            # break

    # for qa, qb, qc, qd in doubleex:
    #     dc = Circuit(n_qubits)
    #     dc.add(gates.CNOT(qc, qd))
    #     dc.add(gates.CNOT(qa, qc))
    #     dc.add(gates.H(qa))
    #     dc.add(gates.H(qd))
    #     dc.add(gates.CNOT(qa, qb))
    #     dc.add(gates.CNOT(qc, qd))
    #     dc.add(gates.RY(qa, theta=-x/8))
    #     dc.add(gates.RY(qb, theta=x/8))
    #     dc.add(gates.CNOT(qa, qd))
    #     dc.add(gates.H(qd))
    #     dc.add(gates.CNOT(qd, qb))
    #     dc.add(gates.RY(qa, theta=-x/8))
    #     dc.add(gates.RY(qb, theta=x/8))
    #     dc.add(gates.CNOT(qc, qb))
    #     dc.add(gates.CNOT(qc, qa))
    #     dc.add(gates.RY(qa, theta=x/8))
    #     dc.add(gates.RY(qb, theta=-x/8))
    #     dc.add(gates.CNOT(qd, qb))
    #     dc.add(gates.H(qd))
    #     dc.add(gates.CNOT(qa, qd))
    #     dc.add(gates.RY(qa, theta=x/8))
    #     dc.add(gates.RY(qb, theta=-x/8))
    #     dc.add(gates.CNOT(qa, qb))
    #     dc.add(gates.CNOT(qc, qa))
    #     dc.add(gates.H(qa))
    #     dc.add(gates.H(qd))
    #     dc.add(gates.CNOT(qa, qc))
    #     dc.add(gates.CNOT(qc, qd))
    #     c += dc
    #     for _i, parameter in enumerate(c.get_parameters()):
    #         gradient = parameter_shift(c, hamiltonian, parameter_index=_i)
    #         print(f"Excitation {qa, qb, qc, qd} => Gradient: {gradient}")
    #         if np.abs(gradient) > 1e-10:
    #             gradients[(qa, qb, qc, qd)] = gradient
    #         break
    return c


# checkexcitation(sum(n_electrons), n_qubits)
print(list(enumerate(c.get_parameters())))
c = build(c, s_excitations, d_excitations, 0.0)
print(c.draw())
print(c.summary())

for excitation in sorted(gradients, key=lambda x: np.abs(gradients[x]), reverse=True):
    print(f"Excitation {excitation} => Gradient: {gradients[excitation]}")

params = []
s_coeffs = [1 / 2, -1 / 2]
d_coeffs = [-1 / 8, 1 / 8, -1 / 8, 1 / 8, 1 / 8, -1 / 8, 1 / 8, -1 / 8]

for i in s_excitations:
    params += s_coeffs

for i in d_excitations:
    params += d_coeffs

## Restricted
x = []


def electronic_energy(x):
    x *= params
    c.set_parameters(x)
    return expectation(c, hamiltonian)


y = []


def gradient_restriction(y):
    y *= params
    c.set_parameters(y)
    return np.array([parameter_shift(c, hamiltonian, parameter_index=_i) for _i in range(len(y))])


print(np.array([parameter_shift(c, hamiltonian, parameter_index=_i) for _i in range(len(y))]))


## Non-Restricted
def nr_electronic_energy(parameter):
    c.set_parameters(parameter)
    return expectation(c, hamiltonian)


y = []


def gradient_no_restriction(y):
    return np.array([parameter_shift(c, hamiltonian, parameter_index=_i) for _i in range(len(y))])


print(np.array([parameter_shift(c, hamiltonian, parameter_index=_i) for _i in range(len(y))]))


## Energy Calculation
methods = ["SLSQP", "BFGS", "COBYLA", "Powell"]

for m in methods:
    theta = np.zeros(len(c.get_parameters()))
    if m in ["BFGS", "SLSQP"]:
        # Gradients for BFGS and SLSQP
        best, params, extra = optimize(electronic_energy, theta, method=m, jac=gradient_restriction)
    else:
        best, params, extra = optimize(electronic_energy, theta, method=m)

    print(f"Energy {m}: {best}")
    print(f"Optimized parameter: {params}")
    print(f"Number of steps: {extra.nfev}")

print(f"Exact result: {hamiltonian.eigenvalues()[0]}")


# LiH
mol = Molecule(xyz_file="lih.xyz")
mol.run_pyscf()
mol.hf_embedding(active=[1, 2, 5])
hamiltonian = mol.hamiltonian(oei=mol.embed_oei, tei=mol.embed_tei, constant=mol.inactive_energy)
n_qubits = mol.n_active_orbs
n_electrons = mol.n_active_e
s_excitations, d_excitations = checkexcitation(n_electrons, n_qubits)
gradients = {}

## Circuit construction

c = hf_circuit(n_qubits, n_electrons)
s_excitations, d_excitations = checkexcitation(n_electrons, n_qubits)
c = build(c, s_excitations, d_excitations, 0.1)
print(c.draw())
print(c.summary())

print(f"\nInitial number of excitation: {len(s_excitations + d_excitations)}")
print(f"Final number of excitations: {len(gradients)}")

for excitation in sorted(gradients, key=lambda x: np.abs(gradients[x]), reverse=True):
    print(f"Excitation {excitation} => Gradient: {gradients[excitation]}")

## Calculation of Hamiltonian

param_range = np.arange(*map(float, input("Enter space-seperated parameter range, and step increment = ").split()))
for x in param_range:
    print(f"Theta: {round(x, 5)} => Energy expectation value: {electronic_energy(x)}")

## Restricted
x = []


def electronic_energy(x):
    x *= params
    c.set_parameters(x)
    return expectation(c, hamiltonian)


# param_range = np.arange(*map(float, input("Enter space-seperated parameter range, and step increment = ").split()))
# for x in param_range:
#  print(f"Theta: {round(x, 5)} => Energy expectation value: {electronic_energy(x)}")

# print(params)

y = []


def gradient_restriction(y):
    y *= params
    c.set_parameters(y)
    return np.array([parameter_shift(c, hamiltonian, parameter_index=_i) for _i in range(len(y))])


print(np.array([parameter_shift(c, hamiltonian, parameter_index=_i) for _i in range(len(y))]))

## Energy Calculation

methods = ["SLSQP", "BFGS", "COBYLA", "Powell"]

for m in methods:
    theta = np.zeros(len(c.get_parameters()))
    if m in ["BFGS", "SLSQP"]:
        # Gradients for BFGS and SLSQP
        best, params, extra = optimize(electronic_energy, theta, method=m, jac=gradient_restriction)
    else:
        best, params, extra = optimize(electronic_energy, theta, method=m)

    print(f"Energy {m}: {best}")
    print(f"Optimized parameter: {params}")
    print(f"Number of steps: {extra.nfev}")

print(f"Exact result: {hamiltonian.eigenvalues()[0]}")
