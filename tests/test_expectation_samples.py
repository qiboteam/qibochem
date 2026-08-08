"""
Test expectation functionality
"""

import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from qibochem.driver import Molecule
from qibochem.measurement import expectation_from_samples, v_expectation
from qibochem.measurement.optimization import _measurement_basis_rotations
from qibochem.measurement.result import (
    _pauli_term_measurement_expectation,
    sample_statistics,
)


@pytest.mark.parametrize(
    "term,frequencies,qubit_map,expected",
    [
        (X(0), {"10": 5}, [0, 1], -1.0),
        (X(2), {"010": 5}, [0, 2, 5], -1.0),
        (Y(4), {"110": 5}, [0, 2, 4], 1.0),
        (X(0) * Y(1), {"11": 5}, [0, 1], 1.0),
        (X(0) * Y(1) + X(0), {"11": 5}, [0, 1], 0.0),
    ],
)
def test_pauli_term_measurement_expectation(term, frequencies, qubit_map, expected):
    result = _pauli_term_measurement_expectation(term, frequencies, qubit_map)
    assert result == expected


def test_measurement_basis_rotations_error():
    """If unknown measurement grouping scheme used"""
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    with pytest.raises(NotImplementedError):
        _ = _measurement_basis_rotations(hamiltonian, grouping="test")


@pytest.mark.parametrize(
    "gates_to_add,shot_allocation,expected",
    [
        ([gates.H(0)], [10, 0], 1.0),  # State vector: 1/sqrt(2)(|0> + |1>), Measuring X
        ([gates.X(0), gates.Z(0)], [0, 10], -1.0),  # State vector: -|1>, Measuring Z
    ],
)
def test_expectation_manual_shot_allocation(gates_to_add, shot_allocation, expected):
    circuit = Circuit(1)
    circuit.add(gates_to_add)
    hamiltonian = SymbolicHamiltonian(X(0) + Z(0))
    result = expectation_from_samples(
        circuit, hamiltonian, n_shots_per_pauli_term=False, shot_allocation=shot_allocation
    )
    assert result == pytest.approx(expected), f"Result {result} != Exact {expected}"


def test_expectation_invalid_shot_allocation():
    circuit = Circuit(1)
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    shot_allocation = (1,)
    with pytest.raises(ValueError):
        _ = expectation_from_samples(
            circuit, hamiltonian, n_shots_per_pauli_term=False, shot_allocation=shot_allocation
        )


@pytest.mark.parametrize("grouping", ["qwc", "gc", "gc2"])
@pytest.mark.parametrize(
    "terms",
    [
        Z(2),
        0.2 * X(0) + Y(2) + 13.0,
        Z(0) + X(0) * Y(1) + Z(0) * Y(2),
        Y(0) + Z(1) + X(0) * Z(2),
        0.1 * X(0) * X(1) * Y(2) + 0.2 * X(0) * Y(1) * Y(2) + 0.3 * Y(0) * X(1) * X(2) - 3.14 * Y(0) * Y(1) * X(2),
    ],
)
def test_measurement_grouping_functionality(grouping, terms):
    """Small scale tests of commuting measurements functionality"""
    nqubits = 3
    circuit = Circuit(nqubits)
    circuit.add(gates.RX(_i, 0.1 * _i) for _i in range(nqubits))
    circuit.add(gates.CNOT(_i, _i + 1) for _i in range(nqubits - 1))
    circuit.add(gates.RZ(_i, 0.2 * _i) for _i in range(nqubits))
    hamiltonian = SymbolicHamiltonian(terms, nqubits=nqubits)
    expected = hamiltonian.expectation(circuit)
    test = expectation_from_samples(
        circuit,
        hamiltonian,
        n_shots=100_000,
        grouping=grouping,
    )
    assert test == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("grouping", ["gc", "gc2"])
@pytest.mark.parametrize(
    "terms,nqubits,gates_to_add",
    [
        (0.5 * X(0) * Y(1) + Z(0) * Z(1) * Z(2), 3, (gates.H(0), gates.RX(1, theta=-np.pi / 2))),
        (
            0.5 * Y(0) * X(1) * Z(3) * Z(4) + Z(0) * Z(1) * X(2) * Z(3) * Z(4),
            5,
            (gates.RX(0, theta=-np.pi / 2), gates.H(1)),
        ),
        (Y(1) * Y(2) + X(0) * X(1) * Z(2), 3, (gates.H(0), gates.H(1))),
        (Y(0) * X(1) + X(0) * Y(1) * Z(2), 3, (gates.H(0), gates.S(0), gates.H(1), gates.X(2), gates.H(2))),
    ],
)
def test_measurement_grouping_extra_tests(grouping, terms, nqubits, gates_to_add):
    """Additional tests for generally commuting terms"""
    hamiltonian = SymbolicHamiltonian(terms, nqubits=nqubits)
    circuit = Circuit(nqubits)
    circuit.add(gates_to_add)
    result = expectation_from_samples(circuit, hamiltonian, n_shots=100_000, grouping=grouping)
    assert result == pytest.approx(hamiltonian.expectation(circuit), abs=0.03)


def test_h2_hf_energy():
    """Test HF energy of H2 molecule"""
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()

    # JW-HF circuit
    circuit = Circuit(4)
    circuit.add(gates.X(_i) for _i in range(2))
    # Molecular Hamiltonian and the HF expectation value
    hamiltonian = h2.hamiltonian()

    n_shots = 50000
    for n_shots_per_pauli_term in (True, False):
        hf_energy = expectation_from_samples(
            circuit,
            hamiltonian,
            n_shots_per_pauli_term=n_shots_per_pauli_term,
            n_shots=n_shots,
            grouping="gc",
        )
        # Hardcoded threshold should be high enough with so many shots
        assert hf_energy == pytest.approx(hamiltonian.expectation(circuit), abs=0.01)


@pytest.mark.parametrize(
    "terms,grouping,expected_means,expected_variances",
    [
        (X(0), None, [1.0], [0.0]),
        (X(0) + Z(0), None, [1.0, 0.0], [0.0, 0.0]),
        (Z(0) + X(0) * Z(1), "qwc", [-1.0, 0.0], [0.0, 0.0]),
    ],
)
def test_sample_statistics(terms, grouping, expected_means, expected_variances):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.X(1))
    n_trial_shots = 20_000
    hamiltonian = SymbolicHamiltonian(terms, nqubits=2)
    grouped_terms = _measurement_basis_rotations(hamiltonian, grouping)
    sample_means, sample_variances = sample_statistics(circuit, grouped_terms, n_shots=n_trial_shots)
    assert sample_means == pytest.approx(expected_means, abs=0.08)
    assert sample_variances == pytest.approx(expected_variances, abs=0.1)


@pytest.mark.parametrize(
    "terms,grouping",
    [
        (0.2 * X(0) + Y(2) + 13.0, None),
        (0.2 * X(0) + Y(2) + 13.0, "qwc"),
        (Z(0) + X(0) * Y(1) + Z(0) * Y(2), None),
        (Y(0) + Z(1) + X(0) * Z(2), "qwc"),
        (Y(0) + Z(1) + X(0) * Z(2), "gc"),
    ],
)
def test_v_expectation_vmsa(terms, grouping):
    """Small scale tests of variance-based expectation value evaluation"""
    nqubits = 3
    circuit = Circuit(nqubits)
    circuit.add(gates.RX(_i, 0.1 * _i) for _i in range(nqubits))
    circuit.add(gates.CNOT(_i, _i + 1) for _i in range(nqubits - 1))
    circuit.add(gates.RZ(_i, 0.2 * _i) for _i in range(nqubits))
    hamiltonian = SymbolicHamiltonian(terms, nqubits=nqubits)
    expected = hamiltonian.expectation(circuit)
    n_shots = 50_000
    n_trial_shots = 2000
    test = v_expectation(
        circuit,
        hamiltonian,
        n_trial_shots=n_trial_shots,
        n_shots=n_shots,
        grouping=grouping,
    )
    assert test == pytest.approx(expected, abs=0.08)
