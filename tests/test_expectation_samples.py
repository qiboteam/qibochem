"""
Test expectation functionality
"""

# import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Z

from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation


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


def test_h2_hf_energy():
    """Test HF energy of H2 molecule"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    try:
        h2.run_pyscf()
    except ModuleNotFoundError:
        h2.run_psi4()

    # JW-HF circuit
    circuit = Circuit(4)
    circuit.add(gates.X(_i) for _i in range(2))

    # Molecular Hamiltonian and the HF expectation value
    hamiltonian = h2.hamiltonian()
    hf_energy = expectation(circuit, hamiltonian, from_samples=True, n_shots=10000)

    assert h2_ref_energy == pytest.approx(hf_energy, abs=0.005)
