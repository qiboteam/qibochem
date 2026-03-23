"""
Test functionality to reduce the measurement cost of running VQE
"""

import numpy as np
import pytest
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from qibochem.measurement.optimization import measurement_basis_rotations
from qibochem.measurement.shot_allocation import (
    allocate_shots,
    allocate_shots_by_variance,
)
from qibochem.measurement.util import (
    check_terms_commutativity,
    group_commuting_terms,
    pauli_to_symplectic,
    symplectic_inner_product,
    symplectic_to_pauli,
)


@pytest.mark.parametrize(
    "term1,term2,qwc_expected,gc_expected",
    [
        ("X0", "Z0", False, False),
        ("X0", "Z1", True, True),
        ("X0 X1", "Y0 Y1", False, True),
        ("X0 Y1", "Y0 Y1", False, False),
    ],
)
def test_check_terms_commutativity(term1, term2, qwc_expected, gc_expected):
    """Do two Pauli strings commute (qubit-wise or generally)?"""
    qwc_result = check_terms_commutativity(term1, term2, qubitwise=True)
    assert qwc_result == qwc_expected
    gc_result = check_terms_commutativity(term1, term2, qubitwise=False)
    assert gc_result == gc_expected


@pytest.mark.parametrize(
    "term_list,qwc_expected,gc_expected",
    [
        (["X0 Z1", "X0", "Z0", "Z0 Z1"], [["X0", "X0 Z1"], ["Z0", "Z0 Z1"]], [["X0", "X0 Z1"], ["Z0", "Z0 Z1"]]),
        (["X0 Y1 Z2", "X1 X2", "Z1 Y2"], [["X0 Y1 Z2"], ["X1 X2"], ["Z1 Y2"]], [["X0 Y1 Z2", "X1 X2", "Z1 Y2"]]),
    ],
)
def test_group_commuting_terms(term_list, qwc_expected, gc_expected):
    qwc_result = group_commuting_terms(term_list, qubitwise=True)
    assert qwc_result == qwc_expected
    gc_result = group_commuting_terms(term_list, qubitwise=False)
    assert gc_result == gc_expected


@pytest.mark.parametrize(
    # "pauli_string,n_qubits,expected",
    "function_args,expected",
    [
        ({"pauli_string": ["X0", "Y1", "Z2"], "n_qubits": 4}, np.array([1, 1, 0, 0, 0, 1, 1, 0])),
        ({"pauli_string": ["Z1", "X3"], "n_qubits": 4}, np.array([0, 0, 0, 1, 0, 1, 0, 0])),
        ({"pauli_string": [], "n_qubits": 4}, np.array([0, 0, 0, 0, 0, 0, 0, 0])),
    ],
)
def test_pauli_to_symplectic(function_args, expected):
    result = pauli_to_symplectic(**function_args)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "function_args,expected",
    [
        ({"symplectic_vector": np.array([1, 1, 0, 0, 0, 1, 1, 0])}, ["X0", "Y1", "Z2"]),
        ({"symplectic_vector": np.array([0, 1, 0, 1, 0, 1, 0, 0])}, ["Y1", "X3"]),
    ],
)
def test_symplectic_to_pauli(function_args, expected):
    result = symplectic_to_pauli(**function_args)
    assert result == expected


@pytest.mark.parametrize(
    "u,v",
    [
        (np.array([1, 1, 0, 0, 0, 1, 1, 0]), np.array([1, 1, 0, 0, 0, 1, 1, 0])),
        (np.array([1, 0, 0, 0, 1, 1, 1, 1]), np.array([1, 1, 0, 0, 0, 1, 1, 0])),
    ],
)
def test_symplectic_inner_product(u, v):
    # Using the actual definition instead of array slicing to calculate the symplectic inner product
    dim = u.shape[0] // 2
    j_matrix = np.concatenate(
        (
            np.concatenate((np.zeros((dim, dim)), np.identity(dim, dtype=int)), axis=1),
            np.concatenate((np.identity(dim, dtype=int), np.zeros((dim, dim))), axis=1),
        ),
        axis=0,
    )
    assert symplectic_inner_product(u, v) == (np.dot(u, np.dot(j_matrix, v)).astype(int) % 2)


@pytest.mark.parametrize(
    "method,max_shots_per_term,expected",
    [
        ("u", None, [66, 67, 67]),  # Control test; i.e. working normally
        (None, None, [23, 168, 9]),  # Default arguments test
        (None, 100, [75, 100, 25]),  # max_shots_per_term error
        (None, 25, [75, 100, 25]),  # If max_shots_per_term is too small
        (None, 1000, [23, 168, 9]),  # If max_shots_per_term is too large
    ],
)
def test_allocate_shots(method, max_shots_per_term, expected):
    hamiltonian = SymbolicHamiltonian(94 * Z(0) + Y(1) + 5 * X(0))  # Note that SymPy sorts the terms as X0 -> Z0 -> Z1
    grouped_terms = measurement_basis_rotations(hamiltonian)
    n_shots = 200
    test_allocation = allocate_shots(
        grouped_terms, method=method, n_shots=n_shots, max_shots_per_term=max_shots_per_term
    )
    # Might have the occasional off by one error, hence set the max allowed difference to be 1
    assert max(abs(_i - _j) for _i, _j in zip(test_allocation, expected)) <= 1


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
    "method,expected",
    [
        ("vmsa", [0, 20, 40]),
        ("vpsr", [0, 12, 24]),
    ],
)
def test_allocate_shots_by_variance(method, expected):
    total_shots = 120
    n_trial_shots = 20
    variance_values = [0.0, 16.0, 64.0]
    test_allocation = allocate_shots_by_variance(total_shots, n_trial_shots, variance_values, method=method)
    assert test_allocation == expected
