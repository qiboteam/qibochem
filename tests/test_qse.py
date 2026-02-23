"""Tests for Quantum Subspace Expansion (QSE)."""

import numpy as np
import pytest
from qibo import models

from qibochem.driver import Molecule
from qibochem.ansatz import hf_circuit
from qibochem.selected_ci.qse import QSE, QSEConfig, qse


def test_qse_h2_hf_reference():
    """Test QSE on H2 molecule using a Hartree-Fock reference state."""
    # Define an H2 molecule
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))], basis="sto-3g")
    h2.run_pyscf()

    # Exact diagonalization to find the ground state energy for reference
    exact_energies = h2.eigenvalues(h2.hamiltonian())
    fci_ground_energy = exact_energies[0]

    # Create HF circuit
    circuit = hf_circuit(h2.nso, h2.nelec)

    # Run QSE
    config = QSEConfig(conserve_spin=True, excitation_threshold=1e-6)
    result = qse(h2, circuit, config=config)

    # The QSE energy should be at least as good as the HF energy
    assert result.energies[0] <= h2.e_hf + 1e-8

    # Ensure it returns the expected number of eigenpairs.
    # For H2 (4 spin-orbitals) starting from HF, we expect a subspace dimension 
    # capturing the ground state and single excitations.
    assert result.subspace_dimension > 0
    assert result.projected_subspace_dimension <= result.subspace_dimension
    assert len(result.energies) == result.projected_subspace_dimension
    assert len(result.energies) > 0
    assert result.eigenvectors.shape == (result.subspace_dimension, len(result.energies))
    assert result.total_circuit_runs == 0


def test_qse_operator_generation():
    """Test that QSE generates proper excitation operators."""
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))])
    h2.run_pyscf()

    # nso = 4 for minimal basis H2
    # If conserve_spin=True, pairs are (even, even) and (odd, odd)
    # (0,0), (0,2), (2,0), (2,2) -- 4
    # (1,1), (1,3), (3,1), (3,3) -- 4
    # + Identity ("") = 9
    config = QSEConfig(conserve_spin=True)
    runner = QSE(h2, config)
    ops = runner._generate_excitation_operators(4)
    assert len(ops) == 9

    # If conserve_spin=False, any combination of i,j is allowed
    # 4 * 4 = 16 + 1 = 17
    config_no_spin = QSEConfig(conserve_spin=False)
    runner_no_spin = QSE(h2, config_no_spin)
    ops_no_spin = runner_no_spin._generate_excitation_operators(4)
    assert len(ops_no_spin) == 17

def test_qse_h2_measured():
    """Test QSE on H2 molecule using hardware-viable measurements."""
    # Define an H2 molecule
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))], basis="sto-3g")
    h2.run_pyscf()

    # Create HF circuit
    circuit = hf_circuit(h2.nso, h2.nelec)

    # Run QSE with finite number of shots to mimic hardware
    config = QSEConfig(conserve_spin=True, excitation_threshold=1e-5, eigenvalue_threshold=1e-5, n_shots=20000)
    result = qse(h2, circuit, config=config)

    # With shots, the energy has noise, so we just check if it loosely bounds e_hf
    assert result.energies[0] <= h2.e_hf + 0.15
    assert result.subspace_dimension > 0
    assert len(result.energies) > 0
    assert result.total_circuit_runs > 0
