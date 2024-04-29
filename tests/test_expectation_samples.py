"""
Test expectation functionality
"""

import pytest
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from qibochem.driver import Molecule
from qibochem.measurement import expectation, expectation_from_samples
from qibochem.measurement.optimization import (
    allocate_shots,
    measurement_basis_rotations,
)
from qibochem.measurement.result import pauli_term_measurement_expectation


@pytest.mark.parametrize(
    "term,frequencies,qubit_map,expected",
    [
        (X(0), {"10": 5}, [0, 1], -1.0),
        (X(2), {"010": 5}, [0, 2, 5], -1.0),
        (Y(4), {"110": 5}, [0, 2, 4], 1.0),
    ],
)
def test_pauli_term_measurement_expectation(term, frequencies, qubit_map, expected):
    symbolic_term = SymbolicHamiltonian(term).terms[0]
    result = pauli_term_measurement_expectation(symbolic_term, frequencies, qubit_map)
    assert result == expected, "{term}"


@pytest.mark.parametrize(
    "terms,gates_to_add,expected",
    [
        (Z(0), [gates.X(0)], -1.0),
        (Z(0) * Z(1), [gates.X(0)], -1.0),
        (X(0), [gates.H(0)], 1.0),
    ],
)
def test_expectation_from_samples(terms, gates_to_add, expected):
    hamiltonian = SymbolicHamiltonian(terms)
    circuit = Circuit(2)
    circuit.add(gates_to_add)
    result = expectation_from_samples(circuit, hamiltonian)
    assert result == expected


def test_measurement_basis_rotations_error():
    """If unknown measurement grouping scheme used"""
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    with pytest.raises(NotImplementedError):
        _ = measurement_basis_rotations(hamiltonian, grouping="test")


@pytest.mark.parametrize(
    "method,max_shots_per_term,expected",
    [
        ("u", None, [100, 100]),  # Control test; i.e. working normally
        (None, None, [190, 10]),  # Default arguments test
        (None, 100, [100, 100]),  # max_shots_per_term error
        (None, 25, [100, 100]),  # If max_shots_per_term is too small
        (None, 1000, [190, 10]),  # If max_shots_per_term is too large
    ],
)
def test_allocate_shots(method, max_shots_per_term, expected):
    hamiltonian = SymbolicHamiltonian(94 * Z(0) + Z(1) + 5 * X(0))
    grouped_terms = measurement_basis_rotations(hamiltonian)
    n_shots = 200
    assert (
        allocate_shots(grouped_terms, method=method, n_shots=n_shots, max_shots_per_term=max_shots_per_term) == expected
    )


def test_allocate_shots_coefficient_edge_case():
    """Edge cases of allocate_shots"""
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    grouped_terms = measurement_basis_rotations(hamiltonian)
    n_shots = 1
    assert allocate_shots(grouped_terms, n_shots=n_shots) in ([0, 1], [1, 0])


def test_allocate_shots_input_validity():
    hamiltonian = SymbolicHamiltonian(94 * Z(0) + Z(1) + 5 * X(0))
    grouped_terms = measurement_basis_rotations(hamiltonian)
    with pytest.raises(NameError):
        _ = allocate_shots(grouped_terms, n_shots=1, method="wrong")


@pytest.mark.parametrize(
    "gates_to_add,shot_allocation,threshold,expected",
    [
        ([gates.X(0), gates.Z(0)], [10, 0], None, -1.0),  # State vector: -|1>, Measuring X
        ([gates.X(0)], [0, 1000], 0.1, 0.0),  # State vector: |1>, Measuring X
    ],
)
def test_expectation_manual_shot_allocation(gates_to_add, shot_allocation, threshold, expected):
    circuit = Circuit(1)
    circuit.add(gates_to_add)
    hamiltonian = SymbolicHamiltonian(Z(0) + X(0))
    result = expectation_from_samples(
        circuit, hamiltonian, n_shots_per_pauli_term=False, shot_allocation=shot_allocation
    )
    assert pytest.approx(result, abs=threshold) == expected


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
        SymbolicHamiltonian(X(0) + Y(2)),
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
    n_shots = 5000
    test = expectation_from_samples(
        circuit,
        hamiltonian,
        n_shots=n_shots,
        group_pauli_terms="qwc",
    )
    assert test == pytest.approx(
        expected, abs=0.05
    ), f"Failure: {[[factor.name for factor in term.factors] for term in hamiltonian.terms]}"


@pytest.mark.parametrize(
    "n_shots_per_pauli_term,threshold",
    [
        (True, 0.005),  # 5000 shots used for each term in Hamiltonian
        (False, 0.02),  # 5000 shots divided between each Pauli string in Hamiltonian
    ],
)
def test_h2_hf_energy(n_shots_per_pauli_term, threshold):
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

    n_shots = 5000
    hf_energy = expectation_from_samples(
        circuit,
        hamiltonian,
        n_shots_per_pauli_term=n_shots_per_pauli_term,
        n_shots=n_shots,
        group_pauli_terms="qwc",
    )
    assert pytest.approx(hf_energy, abs=threshold) == h2_ref_energy
