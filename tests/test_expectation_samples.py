"""
Test expectation functionality
"""

# import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from qibochem.driver import Molecule
from qibochem.measurement import expectation
from qibochem.measurement.optimization import (
    allocate_shots,
    measurement_basis_rotations,
)


def test_expectation_z0():
    """Test from_samples functionality of expectation function"""
    hamiltonian = SymbolicHamiltonian(Z(0))
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    result = expectation(circuit, hamiltonian, from_samples=True)
    assert pytest.approx(result) == -1.0


def test_expectation_z0z1():
    """Tests expectation_from_samples for diagonal Hamiltonians (only Z's)"""
    hamiltonian = SymbolicHamiltonian(Z(0) * Z(1))
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    result = expectation(circuit, hamiltonian, from_samples=True)
    assert pytest.approx(result) == -1.0


def test_expectation_x0():
    """Tests expectation_from_samples for Hamiltonians with X"""
    hamiltonian = SymbolicHamiltonian(X(0))
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    result = expectation(circuit, hamiltonian, from_samples=True)
    assert pytest.approx(result) == 1.0


def test_expectation_x0_2():
    """Test 2 of expectation_from_samples for Hamiltonians with X"""
    hamiltonian = SymbolicHamiltonian(X(0))
    circuit = Circuit(2)
    result = expectation(circuit, hamiltonian, from_samples=True, n_shots=10000)
    assert pytest.approx(result, abs=0.05) == 0.00


def test_measurement_basis_rotations_error():
    """If unknown measurement grouping scheme used"""
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    with pytest.raises(NotImplementedError):
        _ = measurement_basis_rotations(hamiltonian, 2, grouping="test")


def test_allocate_shots_uniform():
    hamiltonian = SymbolicHamiltonian(94 * Z(0) + Z(1) + 5 * X(0))
    grouped_terms = measurement_basis_rotations(hamiltonian, 1)
    n_shots = 200
    assert allocate_shots(grouped_terms, method="u", n_shots=n_shots) == [100, 100]


def test_allocate_shots_coefficient():
    hamiltonian = SymbolicHamiltonian(94 * Z(0) + Z(1) + 5 * X(0))
    grouped_terms = measurement_basis_rotations(hamiltonian, 1)
    n_shots = 200
    # Default arguments
    assert allocate_shots(grouped_terms, n_shots=n_shots) == [190, 10], "Default arguments error!"
    # Reasonable max_shots_per_term test
    assert allocate_shots(grouped_terms, n_shots=n_shots, max_shots_per_term=100) == [
        100,
        100,
    ], "max_shots_per_term error!"
    # Too small max_shots_per_term test
    assert allocate_shots(grouped_terms, n_shots=n_shots, max_shots_per_term=25) == [
        100,
        100,
    ], "max_shots_per_term error: Too small test"
    # Too big max_shots_per_term test
    assert allocate_shots(grouped_terms, n_shots=n_shots, max_shots_per_term=1000) == [
        190,
        10,
    ], "max_shots_per_term error: Too big test"


def test_expectation_manual_shot_allocation():
    # State vector: -|1>
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.Z(0))
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    shot_allocation = (10, 0)
    result = expectation(
        circuit, hamiltonian, from_samples=True, n_shots_per_pauli_term=False, shot_allocation=shot_allocation
    )
    assert pytest.approx(result) == -1.0


def test_expectation_manual_shot_allocation2():
    # State vector: |1>
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    shot_allocation = (0, 500)
    result = expectation(
        circuit, hamiltonian, from_samples=True, n_shots_per_pauli_term=False, shot_allocation=shot_allocation
    )
    assert pytest.approx(result, abs=0.1) == 0.0


def test_expectation_invalid_shot_allocation():
    circuit = Circuit(1)
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    shot_allocation = (1,)
    with pytest.raises(AssertionError):
        _ = expectation(
            circuit, hamiltonian, from_samples=True, n_shots_per_pauli_term=False, shot_allocation=shot_allocation
        )


def test_h2_hf_energy():
    """Test HF energy of H2 molecule"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()

    # JW-HF circuit
    circuit = Circuit(4)
    circuit.add(gates.X(_i) for _i in range(2))

    # Molecular Hamiltonian and the HF expectation value
    hamiltonian = h2.hamiltonian()
    hf_energy = expectation(circuit, hamiltonian, from_samples=True, n_shots=10000)

    assert h2_ref_energy == pytest.approx(hf_energy, abs=0.005)
