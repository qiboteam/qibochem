"""
Example of the basis rotation circuit with H3+ molecule. Starts with the guess wave function from the core Hamiltonian,
and runs the VQE to obtain the HF energy.
"""

import numpy as np
from qibo.models import VQE

from qibochem.ansatz import circuit_ansatz, hf_circuit
from qibochem.driver import Molecule


def guess_mo_coeffs(hcore, overlap):
    """Generate guess coefficients for MOs using the core Hamiltonian (1-electron terms) and overlap integrals"""
    # Symmetric orthogonalization of overlap matrix S using np:
    u, s, vh = np.linalg.svd(overlap)
    a = u @ np.diag(s ** (-0.5)) @ vh

    # Transform guess Fock matrix formed with H_core
    f_p = a.dot(hcore).dot(a)
    # Diagonalize F_p for eigenvalues and eigenvectors
    _e, c_p = np.linalg.eigh(f_p)
    # Transform c_p back into AO basis and return it
    return a.dot(c_p)  # MO coefficients using a H_core guess


def main():
    """Main function"""
    rng = np.random.default_rng()
    # Define molecule and populate
    mol = Molecule(xyz_file="h3p.xyz")
    mol.run_pyscf()

    # Diagonalize H_core to use as the guess wave function
    mol.ca = guess_mo_coeffs(mol.hcore, mol.overlap)

    # Molecular Hamiltonian with the guess OEI/TEI
    hamiltonian = mol.hamiltonian()

    # Check that the hamiltonian with a HF reference ansatz doesn't yield the correct HF energy
    circuit = hf_circuit(mol.nso, mol.nelec)
    print(
        f"Electronic energy: {hamiltonian.expectation(circuit):.8f} (From the H_core guess)"
    )
    print(f"        HF energy: {mol.e_hf:.8f} (Hartree-Fock energy from PySCF)")

    circuit += circuit_ansatz(mol, ansatz="br", include_hf=False)
    qubit_parameters = rng.random(len(circuit.get_parameters()))
    vqe = VQE(circuit, hamiltonian)
    vqe_result = vqe.minimize(qubit_parameters)

    print(f"VQE energy: {vqe_result[0]:.8f} (Basis rotation ansatz)")
    print()
    print("Optimized qubit parameters:\n", vqe_result[1])
    print("Optimizer message:\n", vqe_result[2])


if __name__ == "__main__":
    main()
