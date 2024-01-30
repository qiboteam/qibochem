"""
Example of the basis rotation circuit with H3+ molecule. Starts with the guess wave function from the core Hamiltonian,
    and runs the VQE to obtain the HF energy.
"""

import numpy as np
from qibo.optimizers import optimize
from scipy.optimize import minimize

from qibochem.ansatz.basis_rotation import br_circuit
from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation

# Define molecule and populate
mol = Molecule(xyz_file="h3p.xyz")
try:
    mol.run_pyscf()
except ModuleNotFoundError:
    mol.run_psi4()


# Diagonalize H_core to use as the guess wave function

# First, symmetric orthogonalization of overlap matrix S using np:
u, s, vh = np.linalg.svd(mol.overlap)
A = u @ np.diag(s ** (-0.5)) @ vh

# Transform guess Fock matrix formed with H_core
F_p = A.dot(mol.hcore).dot(A)
# Diagonalize F_p for eigenvalues and eigenvectors
_e, C_p = np.linalg.eigh(F_p)
# Transform C_p back into AO basis
C = A.dot(C_p)
# Form OEI/TEI with the (guess) MO coefficients
oei = np.einsum("pQ, pP -> PQ", np.einsum("pq, qQ -> pQ", mol.hcore, C, optimize=True), C, optimize=True)
# TEI code from https://pycrawfordprogproj.readthedocs.io/en/latest/Project_04/Project_04.html
tei = np.einsum("up, vq, uvkl, kr, ls -> prsq", C, C, mol.aoeri, C, C, optimize=True)

# Molecular Hamiltonian with the guess OEI/TEI
hamiltonian = mol.hamiltonian(oei=oei, tei=tei)

# Check that the hamiltonian with a HF reference ansatz doesn't yield the correct HF energy
circuit = hf_circuit(mol.nso, sum(mol.nelec))
print(
    f"\nElectronic energy: {expectation(circuit, hamiltonian):.8f} (From the H_core guess, should be > actual HF energy)"
)


def electronic_energy(parameters):
    """
    Loss function (Electronic energy) for the basis rotation ansatz
    """
    circuit = hf_circuit(mol.nso, sum(mol.nelec))
    circuit += br_circuit(mol.nso, parameters, sum(mol.nelec))

    return expectation(circuit, hamiltonian)


# Build  a basis rotation_circuit
params = np.random.rand(sum(mol.nelec) * (mol.nso - sum(mol.nelec)))  # n_occ * n_virt

best, params, extra = optimize(electronic_energy, params)

print("\nResults using Qibo optimize:")
print(f" HF energy: {mol.e_hf:.8f}")
print(f"VQE energy: {best:.8f} (Basis rotation ansatz)")
# print()
# print("Optimized parameters:", params)


params = np.random.rand(sum(mol.nelec) * (mol.nso - sum(mol.nelec)))  # n_occ * n_virt

result = minimize(electronic_energy, params)
best, params = result.fun, result.x

print("\nResults using scipy.optimize:")
print(f" HF energy: {mol.e_hf:.8f}")
print(f"VQE energy: {best:.8f} (Basis rotation ansatz)")
# print()
# print("Optimized parameters:", params)
