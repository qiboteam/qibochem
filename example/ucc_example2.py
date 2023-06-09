"""
Sample script of how to build and run the UCCSD ansatz with HF embedding for LiH.
This example uses some utility functions to build the UCCSD ansatz; should be easier to use

TODO: IN-PROGRESS!!!
"""

import numpy as np

from scipy.optimize import minimize
from qibo.optimizers import optimize

from qibochem.driver.molecule import Molecule

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.ucc import ucc_ansatz


# Define molecule and populate
mol = Molecule(xyz_file="lih.xyz")
try:
    mol.run_pyscf()
except ModuleNotFoundError:
    mol.run_psi4()


# Apply embedding and boson encoding
mol.hf_embedding(active=[1, 2, 5])
hamiltonian = mol.hamiltonian(
    oei=mol.embed_oei, tei=mol.embed_tei, constant=mol.inactive_energy
)

# Set parameters for the rest of the experiment
n_qubits = mol.n_active_orbs
n_electrons = mol.n_active_e

# Build circuit
circuit = hf_circuit(n_qubits, n_electrons)

circuit += ucc_ansatz(mol)

# Draw the circuit?
# print(circuit.draw())
# print()


def electronic_energy(parameters):
    """
    Loss function for the circuit ansatz

    TODO: IN-PROGRESS; Will probably be written up as another utility function in ucc.py?
    """
    circuit.set_parameters(parameters)

    return mol.expectation(circuit, hamiltonian)


# Reference energy
fci_energy = hamiltonian.eigenvalues()[0]

n_parameters = len(circuit.get_parameters())

# Random initialization
params = np.random.rand(n_parameters)

# NOTE: Circuit parameters not restricted to be equal within a single UCC excitation!

best, params, extra = optimize(electronic_energy, params)

print("\nResults using Qibo optimize:")
print(f"FCI energy: {fci_energy:.8f}")
print(f" HF energy: {mol.e_hf:.8f} (Classical)")
print(f"VQE energy: {best:.8f} (UCCSD ansatz)")
print()
print("Optimized parameters:", params)


# Scipy minimize
params = np.random.rand(n_parameters)

result = minimize(electronic_energy, params)
best, params = result.fun, result.x

print("\nResults using scipy.optimize:")
print(f"FCI energy: {fci_energy:.8f}")
print(f" HF energy: {mol.e_hf:.8f} (Classical)")
print(f"VQE energy: {best:.8f} (UCCSD ansatz)")
print()
print("Optimized parameters:", params)


full_ham = mol.hamiltonian("f")
mol_fci_energy = mol.eigenvalues(full_ham)[0]

print(f"\nFCI energy: {mol_fci_energy:.8f} (Full Hamiltonian)")
