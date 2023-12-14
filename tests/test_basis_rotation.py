"""
Test for basis rotation ansatz
"""

import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.optimizers import optimize

from qibochem.ansatz.basis_rotation import br_circuit
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation


def test_swap_matrices():
    """Test for swap_matrices"""
    permutations = [[(0, 1), (2, 3)]]
    n_qubits = 4
    smat = basis_rotation.swap_matrices(permutations, n_qubits)
    ref_smat = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])

    assert np.allclose(smat, ref_smat)


def test_unitary_rot_matrix():
    """Test for unitary rotation matrix"""
    occ = [0]
    vir = [1, 2]
    parameters = np.zeros(len(occ) * len(vir))
    parameters += 0.1
    u = basis_rotation.unitary_rot_matrix(parameters, occ, vir)
    ref_u = np.array(
        [[0.99001666, 0.099667, 0.099667], [-0.099667, 0.99500833, -0.00499167], [-0.099667, -0.00499167, 0.99500833]]
    )

    assert np.allclose(u, ref_u)


def test_br_ansatz():
    """Test of basis rotation ansatz against hardcoded HF energies"""
    h2_ref_energy = -1.117349035

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf(max_scf_cycles=1)
    # Use an un-converged wave function to build the Hamiltonian
    mol_sym_ham = mol.hamiltonian()

    # Define quantum circuit
    circuit = Circuit(mol.nso)
    circuit.add(gates.X(_i) for _i in range(sum(mol.nelec)))

    # Add basis rotation ansatz
    # Initialize all at zero
    parameters = np.zeros(sum(mol.nelec) * (mol.nso - sum(mol.nelec)))  # n_occ * n_virt
    circuit += br_circuit(mol.nso, parameters, sum(mol.nelec))

    def electronic_energy(parameters):
        """
        Loss function (Electronic energy) for the basis rotation ansatz
        """
        circuit.set_parameters(parameters)
        return expectation(circuit, mol_sym_ham)

    hf_energy, parameters, _extra = optimize(electronic_energy, parameters)

    assert hf_energy == pytest.approx(h2_ref_energy)
