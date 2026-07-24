"""
Test functionality to reduce the measurement cost of running VQE
"""

import numpy as np
import pytest

from qibochem.measurement.util import (
    _binary_gaussian_elimination,
    _binary_nullspace,
    _check_terms_commutativity,
    _col_reduce_x_matrix,
    _get_sigma_terms,
    _group_commuting_terms,
    _lagrangian_subspace,
    _pauli_to_symplectic,
    _phase_factor,
    _sort_tau_terms,
    _symplectic_inner_product,
    _symplectic_to_pauli,
    _zero_z_matrix,
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
    qwc_result = _check_terms_commutativity(term1, term2, qubitwise=True)
    assert qwc_result == qwc_expected
    gc_result = _check_terms_commutativity(term1, term2, qubitwise=False)
    assert gc_result == gc_expected


@pytest.mark.parametrize(
    "term_list,qwc_expected,gc_expected",
    [
        (["X0 Z1", "X0", "Z0", "Z0 Z1"], [["X0", "X0 Z1"], ["Z0", "Z0 Z1"]], [["X0", "X0 Z1"], ["Z0", "Z0 Z1"]]),
        (["X0 Y1 Z2", "X1 X2", "Z1 Y2"], [["X0 Y1 Z2"], ["X1 X2"], ["Z1 Y2"]], [["X0 Y1 Z2", "X1 X2", "Z1 Y2"]]),
    ],
)
def test_group_commuting_terms(term_list, qwc_expected, gc_expected):
    qwc_result = _group_commuting_terms(term_list, qubitwise=True)
    assert qwc_result == qwc_expected
    gc_result = _group_commuting_terms(term_list, qubitwise=False)
    assert gc_result == gc_expected


@pytest.mark.parametrize(
    # "pauli_string,n_qubits,expected",
    "function_args,expected",
    [
        ({"pauli_string": ["X0", "Y1", "Z2"], "nqubits": 4}, np.array([1, 1, 0, 0, 0, 1, 1, 0])),
        ({"pauli_string": ["Z1", "X3"], "nqubits": 4}, np.array([0, 0, 0, 1, 0, 1, 0, 0])),
        ({"pauli_string": [], "nqubits": 4}, np.array([0, 0, 0, 0, 0, 0, 0, 0])),
    ],
)
def test_pauli_to_symplectic(function_args, expected):
    result = _pauli_to_symplectic(**function_args)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "function_args,expected",
    [
        ({"symplectic_vector": np.array([1, 1, 0, 0, 0, 1, 1, 0])}, ["X0", "Y1", "Z2"]),
        ({"symplectic_vector": np.array([0, 1, 0, 1, 0, 1, 0, 0])}, ["Y1", "X3"]),
    ],
)
def test_symplectic_to_pauli(function_args, expected):
    result = _symplectic_to_pauli(**function_args)
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
    assert _symplectic_inner_product(u, v) == (np.dot(u, np.dot(j_matrix, v)).astype(int) % 2)


@pytest.mark.parametrize(
    "test,result",
    [
        (
            np.array(
                [[0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1]]
            ),
            np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1]]),
        ),
        (
            np.array(
                [[1, 1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]]
            ),
            np.array([[1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]]),
        ),
        (
            np.array([[0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 1]]),
            np.array([[0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]]),
        ),
        (
            np.array([[0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0]]),
            np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0]]),
        ),
        (
            np.array(
                [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 0, 0, 1]]
            ),
            np.array(
                [[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1]]
            ),
        ),
    ],
)
def test_binary_gaussian_elimination(test, result):
    # Hardcoded test results
    test = _binary_gaussian_elimination(test)
    assert np.allclose(test, result), f"RREF forms don't match: {test} != {result}"


def test_binary_nullspace():
    test_space = np.array([[1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]])
    nullspace = _binary_nullspace(test_space)
    assert all(np.allclose((test_space @ vector) % 2, np.zeros(test_space.shape[0])) for vector in nullspace)


def test_lagrangian_subspace():
    # Null space of the test space in test_binary_nullspace
    test_space = np.array(
        [
            [1, 0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    subspace = _lagrangian_subspace(test_space)
    # Vectors in subspace should all be symplectically orthogonal to each other
    assert all(
        _symplectic_inner_product(_v1, _v2) == 0
        for _i1, _v1 in enumerate(subspace)
        for _i2, _v2 in enumerate(subspace)
        if _i1 >= _i2
    )


def test_sort_tau_terms():
    # Using the example given in the function docstring
    test_terms = (["X0", "X2"], ["Z1", "X3", "Z4", "X5"], ["Z0", "Z2"], ["Z1"], ["Z3", "Z5"], ["Z4"])
    test_symplectic_form = np.array([_pauli_to_symplectic(term, nqubits=6) for term in test_terms])
    result_symplectic = _sort_tau_terms(test_symplectic_form)
    result_pauli = [_symplectic_to_pauli(term) for term in result_symplectic]
    assert result_pauli == [["X0", "X2"], ["Z1"], ["Z0", "Z2"], ["Z3", "Z5"], ["Z4"], ["Z1", "X3", "Z4", "X5"]]


def test_get_sigma_terms():
    test_terms = [["X0", "X2"], ["Z1"], ["Z0", "Z2"], ["Z3", "Z5"], ["Z4"], ["Z1", "X3", "Z4", "X5"]]
    test_symplectic_form = [_pauli_to_symplectic(term, nqubits=6) for term in test_terms]
    new_tau_terms, sigma_terms = _get_sigma_terms(test_symplectic_form)
    # Check new tau terms are still mutually orthogonal
    assert all(
        _symplectic_inner_product(_v1, _v2) == 0
        for _i1, _v1 in enumerate(new_tau_terms)
        for _i2, _v2 in enumerate(new_tau_terms)
        if _i1 >= _i2
    )
    # Check sigma terms are also mutually orthogonal
    assert all(
        _symplectic_inner_product(_v1, _v2) == 0
        for _i1, _v1 in enumerate(sigma_terms)
        for _i2, _v2 in enumerate(sigma_terms)
        if _i1 >= _i2
    )
    # Check product of sigma and tau terms
    assert all(
        _symplectic_inner_product(_v1, _v2) == (0 if _i1 > _i2 else 1)
        for _i1, _v1 in enumerate(sigma_terms)
        for _i2, _v2 in enumerate(new_tau_terms)
        if _i1 >= _i2
    )


@pytest.mark.parametrize(
    "vector_space,expected",
    [
        (
            [
                _pauli_to_symplectic(["Z0"], 2),
            ],
            1,
        ),
        (
            [
                _pauli_to_symplectic(["X0", "X1"], 2),
            ],
            1,
        ),
        ([_pauli_to_symplectic(pauli, 2) for pauli in (["X0", "X1"], ["Y0", "Y1"])], -1),
        ([_pauli_to_symplectic(pauli, 3) for pauli in (["X0"], ["Y1", "Z2"], ["Z1", "Y2"])], 1),
    ],
)
def test_phase_factor(vector_space, expected):
    result = _phase_factor(vector_space)
    assert result == expected


def test_col_reduce_x_matrix():
    stabiliser_matrix = np.array(
        [
            [1, 1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 0],
        ]
    )
    gates_list = _col_reduce_x_matrix(stabiliser_matrix)
    # Single column operation, should have only CNOT gate
    assert len(gates_list) == 1 and gates_list[0].name == "cx"


def test_zero_z_matrix():
    stabiliser_matrix = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    gates_list = _zero_z_matrix(stabiliser_matrix)
    # Single column operation, should have only CNOT gate
    assert len(gates_list) == 1 and gates_list[0].name == "sdg"
