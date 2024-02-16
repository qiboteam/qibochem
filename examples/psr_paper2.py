"""
This script is a continuation of 'paper2.py'. After implementing the circuit ansatz, this script attempts to examine the
use of the Parameter Shift Rule implemented in Qibo for chemistry applications. In addition, simulations of LiH are used
for some benchmarking.

Reference paper: https://doi.org/10.1038/s41534-019-0240-1

Acknowledgements: The original draft of this code, in the form of a Jupyter notebook, was prepared by Caleb Seow from
Eunoia Junior College, who was attached to IHPC in December 2023 under the A*STAR Research Attachment Programme for
Junior College students.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from qibo import Circuit, gates, models
from qibo.derivative import parameter_shift
from qibo.optimizers import optimize
from scipy.optimize import minimize

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.ucc import ucc_circuit
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation

# mol = Molecule(xyz_file="h2.xyz")
h2 = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
h2.run_pyscf()

h2.hf_embedding(active=[1, 2, 5])

print(f"# electrons: {h2.n_active_e}")
print(f"# qubits: {h2.n_active_orbs}")

# circuit = hf_circuit(h2.n_active_orbs, h2.n_active_e)
# print(circuit.draw())
# print(h2.eps)
# quit()

hamiltonian = h2.hamiltonian(oei=h2.embed_oei, tei=h2.embed_tei, constant=h2.inactive_energy)

# Exact_result = hamiltonian.eigenvalues()[0]
# print(f"Exact result: {Exact_result}")
# quit()

theta, phi = 0.0, 0.0


def special_R(a, theta, phi):
    A = Circuit(h2.n_active_orbs)
    A.add(gates.RY(a, theta))
    A.add(gates.RZ(a, phi))
    return A


def A_gate(a):
    A = Circuit(h2.n_active_orbs)
    A.add(gates.CNOT(a + 1, a))
    A += special_R(a + 1, theta, phi).invert()
    A.add(gates.CNOT(a, a + 1))
    A += special_R(a + 1, theta, phi)
    A.add(gates.CNOT(a + 1, a))
    return A


# print(circuit.draw())
# print(A_gate(0).get_parameters()) # 4 gradients

# initial state
if h2.n_active_e <= math.ceil(h2.n_active_orbs):
    Xpositions = [2 * i for i in range(h2.n_active_e)]
else:
    Xpositions = [2 * i for i in range(math.ceil(h2.n_active_orbs / 2))]
    Xpositions += [2 * j + 1 for j in range(h2.n_active_e - math.ceil(h2.n_active_orbs / 2))]
    Xpositions.sort()


# setting the circuit up
def setup_electrons(g2):
    for i in Xpositions:
        g2.add(gates.X(i))


Xpositions = [0, 2]

print("Xpositions: ", Xpositions)

# Xpositions is like a starting point
no_of_gates = 0
var_list = []
master_list = []
ref_list = []
popping_list = []

# initialising
for i in range(h2.n_active_orbs):
    if i + 1 not in Xpositions:
        if i != h2.n_active_orbs - 1:
            ref_list.append((i, i + 1))
            master_list.append((i, i + 1))
            no_of_gates += 1

print(f"initial master list: {master_list}")


# Adding 1 layer based on the prev layer
while no_of_gates < math.comb(h2.n_active_orbs, h2.n_active_e):

    for i, j in ref_list:
        var_list.append((i - 1, i))
        var_list.append((j, j + 1))

    var_list.append((0, 0))  # dummy tuple that won't appear naturally

    for i in range(len(var_list) - 1):  # 3 conditions to pop
        if var_list[i][0] < 0:
            popping_list.append(i)
        elif var_list[i][1] > h2.n_active_orbs - 1:
            popping_list.append(i)
        elif var_list[i] == var_list[i + 1]:
            popping_list.append(i + 1)

    popping_list.reverse()  # so that index won't change after every pop

    for i in popping_list:
        var_list.pop(i)

    var_list.pop()  # removing the (0,0) tuple

    for i in var_list:
        master_list.append(i)
        no_of_gates += 1

    ref_list.clear()
    for i in var_list:
        ref_list.append(i)

    popping_list.clear()
    var_list.clear()

print(f"final master list: {master_list}")

A_gate_indexes = master_list[: math.comb(h2.n_active_orbs, h2.n_active_e)]  # shouldnt be needed but jic??

print(f"A gate indexes:    {A_gate_indexes}")

g2 = Circuit(h2.n_active_orbs)

setup_electrons(g2)

for i, j in A_gate_indexes:
    g2 += A_gate(i)

print(g2.draw())

t, p = 0.1, 0.1  # theta, phi
params = []

# qn: for the dagger do we want the changed params too?
# A_gate_param = [-p-np.pi,-t-np.pi/2,t+np.pi/2,p+np.pi]
# A_gate_param = [p+np.pi,t+np.pi/2,t+np.pi/2,p+np.pi]
A_gate_param = [-t, -p, p, t]

for i in range(len(A_gate_indexes)):
    params += A_gate_param

g2.set_parameters(params)

# gradlist = []
#
# gradients = {}
#
# print(len(g2.get_parameters()))
#
# thetalist = [f"theta {i}" for i in range(len(g2.get_parameters()))]
#
# for _i, parameter in enumerate(g2.get_parameters()):
#         gradient = parameter_shift(g2, hamiltonian, parameter_index=_i)
#         gradlist.append(gradient)
#
# for i,j in enumerate(thetalist):
#     if np.abs(gradlist[i]) > 1e-10:
#         gradients[j] = gradient

# print(gradients)

energy = expectation(g2, hamiltonian)
print(f"Expectation: {energy}")

# vqe = models.VQE(g2, hamiltonian)

# Methods: BFGS, COBYLA, Powell
# best_BFGS, params_BFGS, extra_BFGS = vqe.minimize(params, method='BFGS', compile=False)
# print(f"Energy BFGS: {best_BFGS}")
#
# best_COBYLA, params_COBYLA, extra_COBYLA = vqe.minimize(params, method='COBYLA', compile=False)
# print(f"Energy COBYLA: {best_COBYLA}")
#
# best_Powell, params_Powell, extra_Powell = vqe.minimize(params, method='Powell', compile=False)
# print(f"Energy Powell: {best_Powell}")

# Exact_result = hamiltonian.eigenvalues()[0]
# print(f"Exact result: {Exact_result}")


def no_restriction_electronic_energy(parameters):
    g2.set_parameters(parameters)
    return expectation(g2, hamiltonian)


# Combine the gradients of each parameterised gate into a single array (vector):
def gradient_no_restriction(_thetas):
    g2.set_parameters(_thetas)
    return np.array([parameter_shift(g2, hamiltonian, parameter_index=_i) for _i in range(len(_thetas))])


# Start from zero
thetas = np.zeros(len(g2.get_parameters()))

best, params, extra = optimize(no_restriction_electronic_energy, thetas, method="SLSQP", jac=gradient_no_restriction)
# best, params, extra = vqe.minimize(thetas, method='BFGS', jac=ucc_gradient_no_restriction)

print(f"VQE result: {best}")
print(f"Optimized parameter: {params}")
print(f"Number of steps: {extra.nfev}")
print()

# COBYLA
best, params, extra = optimize(no_restriction_electronic_energy, thetas, method="SLSQP")
print(f"VQE result: {best}")
print(f"Optimized parameter: {params}")
print(f"Number of steps: {extra.nfev}")
