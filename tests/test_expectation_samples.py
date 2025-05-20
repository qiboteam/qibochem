"""
Test expectation functionality
"""

import pytest
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from qibochem.driver import Molecule
from qibochem.measurement import expectation, expectation_from_samples, v_expectation
from qibochem.measurement.optimization import measurement_basis_rotations
from qibochem.measurement.result import (
    expectation_variance,
    pauli_term_measurement_expectation,
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
    result = pauli_term_measurement_expectation(term, frequencies, qubit_map)
    assert result == expected


@pytest.mark.parametrize(
    "terms,gates_to_add",
    [
        (Z(0), [gates.X(0)]),
        (Z(0) * Z(1), [gates.X(0)]),
        (X(0), [gates.H(0)]),
    ],
)
def test_expectation_from_samples(terms, gates_to_add):
    hamiltonian = SymbolicHamiltonian(terms, nqubits=2)
    circuit = Circuit(2)
    circuit.add(gates_to_add)
    result = expectation_from_samples(circuit, hamiltonian)
    assert result == pytest.approx(expected := expectation(circuit, hamiltonian)), f"{result} != {expected}"


def test_measurement_basis_rotations_error():
    """If unknown measurement grouping scheme used"""
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    with pytest.raises(NotImplementedError):
        _ = measurement_basis_rotations(hamiltonian, grouping="test")


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
    with pytest.raises(AssertionError):
        _ = expectation_from_samples(
            circuit, hamiltonian, n_shots_per_pauli_term=False, shot_allocation=shot_allocation
        )


@pytest.mark.parametrize(
    "hamiltonian",
    [
        SymbolicHamiltonian(Z(2)),
        SymbolicHamiltonian(0.2 * X(0) + Y(2) + 13.0),
        SymbolicHamiltonian(Z(0) + X(0) * Y(1) + Z(0) * Y(2)),
        SymbolicHamiltonian(Y(0) + Z(1) + X(0) * Z(2)),
    ],
)
def test_qwc_functionality(hamiltonian):
    """Small scale tests of QWC functionality"""
    n_qubits = 3
    circuit = Circuit(n_qubits)
    circuit.add(gates.RX(_i, 0.1 * _i) for _i in range(n_qubits))
    circuit.add(gates.CNOT(_i, _i + 1) for _i in range(n_qubits - 1))
    circuit.add(gates.RZ(_i, 0.2 * _i) for _i in range(n_qubits))
    expected = expectation(circuit, hamiltonian)
    n_shots = 10000
    test = expectation_from_samples(
        circuit,
        hamiltonian,
        n_shots=n_shots,
        grouping="qwc",
    )
    assert test == pytest.approx(expected, abs=0.08)


@pytest.mark.parametrize(
    "n_shots_per_pauli_term,threshold",
    [
        (True, 0.005),  # 5000 shots used for each term in Hamiltonian
        (False, 0.02),  # 5000 shots divided between each Pauli string in Hamiltonian
    ],
)
def test_h2_hf_energy(n_shots_per_pauli_term, threshold):
    """Test HF energy of H2 molecule"""
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()

    # JW-HF circuit
    circuit = Circuit(4)
    circuit.add(gates.X(_i) for _i in range(2))
    # Molecular Hamiltonian and the HF expectation value
    hamiltonian = h2.hamiltonian()

    n_shots = 5000
    hf_energy = expectation_from_samples(
        circuit,
        hamiltonian,
        n_shots_per_pauli_term=n_shots_per_pauli_term,
        n_shots=n_shots,
        grouping="qwc",
    )
    assert hf_energy == pytest.approx(expectation(circuit, hamiltonian), abs=threshold)


@pytest.mark.parametrize(
    "hamiltonian,grouping,expected_variance",
    [
        (SymbolicHamiltonian(X(0), nqubits=2), None, 0.0),
        (SymbolicHamiltonian(X(0) + Z(0), nqubits=2), None, 1.0),
        (SymbolicHamiltonian(Z(0) + X(0) * Z(1), nqubits=2), "qwc", 1.0),
    ],
)
def test_expectation_variance(hamiltonian, grouping, expected_variance):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.X(1))
    n_trial_shots = 100
    sample_mean, sample_variance = expectation_variance(circuit, hamiltonian, n_trial_shots, grouping)
    assert sample_mean == pytest.approx(expectation(circuit, hamiltonian), abs=0.4)
    assert sample_variance == pytest.approx(expected_variance, abs=0.5)


@pytest.mark.parametrize(
    "hamiltonian,grouping",
    [
        (SymbolicHamiltonian(0.2 * X(0) + Y(2) + 13.0), None),
        (SymbolicHamiltonian(0.2 * X(0) + Y(2) + 13.0), "qwc"),
        (SymbolicHamiltonian(Z(0) + X(0) * Y(1) + Z(0) * Y(2)), None),
        (SymbolicHamiltonian(Y(0) + Z(1) + X(0) * Z(2)), "qwc"),
    ],
)
def test_v_expectation_vmsa(hamiltonian, grouping):
    """Small scale tests of variance-based expectation value evaluation"""
    n_qubits = 3
    circuit = Circuit(n_qubits)
    circuit.add(gates.RX(_i, 0.1 * _i) for _i in range(n_qubits))
    circuit.add(gates.CNOT(_i, _i + 1) for _i in range(n_qubits - 1))
    circuit.add(gates.RZ(_i, 0.2 * _i) for _i in range(n_qubits))
    expected = expectation(circuit, hamiltonian)
    n_shots = 1000
    n_trial_shots = 100
    test = v_expectation(
        circuit,
        hamiltonian,
        n_trial_shots=n_trial_shots,
        n_shots=n_shots,
        grouping=grouping,
    )
    assert test == pytest.approx(expected, abs=0.5)
