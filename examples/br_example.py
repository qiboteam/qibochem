"""
Example of the basis rotation circuit with H3+ molecule. Starts with the guess wave function from the core Hamiltonian,
and runs the VQE to obtain the HF energy.
"""

import numpy as np
from qibo.models import VQE

from qibochem.ansatz import basis_rotation, hf_circuit
from qibochem.driver import Molecule


def guess_mo_coeffs(hcore, overlap):
    """Generate guess coefficients for MOs using the core Hamiltonian (1-electron terms) and overlap integrals"""
    # Symmetric orthogonalization of overlap matrix S using np:
    u, s, vh = np.linalg.svd(overlap)
    A = u @ np.diag(s ** (-0.5)) @ vh

    # Transform guess Fock matrix formed with H_core
    F_p = A.dot(hcore).dot(A)
    # Diagonalize F_p for eigenvalues and eigenvectors
    _e, C_p = np.linalg.eigh(F_p)
    # Transform C_p back into AO basis
    mo_coeff = A.dot(C_p)  # MO coefficients using a H_core guess
    return mo_coeff


def basis_rotation_circuit(mol, parameters=0.0):
    """Construct the basis rotation circuit"""
    nqubits = mol.nso
    occ = range(0, mol.nelec)
    vir = range(mol.nelec, mol.nso)

    U, theta = basis_rotation.unitary(occ, vir, parameters=parameters)

    gate_angles, _final_U = basis_rotation.givens_qr_decompose(U)
    gate_layout = basis_rotation.basis_rotation_layout(nqubits)
    gate_list, _ordered_angles = basis_rotation.basis_rotation_gates(gate_layout, gate_angles, theta)

    circuit = hf_circuit(nqubits, mol.nelec)
    circuit.add(gate_list)

    return circuit, gate_angles


def main():
    """Main function"""
    # Define molecule and populate
    mol = Molecule(xyz_file="h3p.xyz")
    mol.run_pyscf()

    # Diagonalize H_core to use as the guess wave function
    mol.ca = guess_mo_coeffs(mol.hcore, mol.overlap)

    # Molecular Hamiltonian with the guess OEI/TEI
    hamiltonian = mol.hamiltonian()

    # Check that the hamiltonian with a HF reference ansatz doesn't yield the correct HF energy
    circuit = hf_circuit(mol.nso, mol.nelec)
    print(f"Electronic energy: {hamiltonian.expectation(circuit):.8f} (From the H_core guess)")
    print(f"        HF energy: {mol.e_hf:.8f} (Hartree-Fock energy from PySCF)")

    br_circuit, qubit_parameters = basis_rotation_circuit(mol, parameters=0.1)
    vqe = VQE(br_circuit, hamiltonian)
    vqe_result = vqe.minimize(qubit_parameters)

    print(f"VQE energy: {vqe_result[0]:.8f} (Basis rotation ansatz)")
    print()
    print("Optimized qubit parameters:\n", vqe_result[1])
    print("Optimizer message:\n", vqe_result[2])


if __name__ == "__main__":
    main()
