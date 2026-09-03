"""
Test functionality to reduce the measurement cost of running VQE
"""

import numpy as np
import pytest
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import I, X, Y, Z
from sympy import srepr

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
    "expression,nqubits,expected",
    [
        (X(0) * Y(1) * Z(2), 4, np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8)),
        (Z(1) * X(3), 4, np.array([0, 0, 0, 1, 0, 1, 0, 0], dtype=np.uint8)),
        (I(2), 4, np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)),
    ],
)
def test_pauli_to_symplectic(expression, nqubits, expected):
    result = _pauli_to_symplectic(expression, nqubits)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "u,v",
    [
        (np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8), np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8)),
        (np.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8), np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8)),
    ],
)
def test_symplectic_inner_product(u, v):
    # Using the actual definition instead of array slicing to calculate the symplectic inner product
    dim = u.shape[0] // 2
    j_matrix = np.concatenate(
        (
            np.concatenate((np.zeros((dim, dim), dtype=np.uint8), np.identity(dim, dtype=np.uint8)), axis=1),
            np.concatenate((np.identity(dim, dtype=np.uint8), np.zeros((dim, dim), dtype=np.uint8)), axis=1),
        ),
        axis=0,
    )
    assert _symplectic_inner_product(u, v) == (np.dot(u, np.dot(j_matrix, v)) % 2)


@pytest.mark.parametrize(
    "term1,term2,qwc_expected,gc_expected",
    [
        (_pauli_to_symplectic(X(0), 1), _pauli_to_symplectic(Z(0), 2), False, False),
        (_pauli_to_symplectic(X(0), 1), _pauli_to_symplectic(Z(1), 3), True, True),
        (_pauli_to_symplectic(X(0) * X(1), 3), _pauli_to_symplectic(Y(0) * Y(1), 2), False, True),
        (_pauli_to_symplectic(X(0) * Y(1), 2), _pauli_to_symplectic(Y(0) * Y(1), 2), False, False),
    ],
)
def test_check_terms_commutativity(term1, term2, qwc_expected, gc_expected):
    """Do two Pauli strings commute (qubit-wise or generally)?"""
    qwc_result = _check_terms_commutativity(term1, term2, qubitwise=True)
    assert qwc_result == qwc_expected
    gc_result = _check_terms_commutativity(term1, term2, qubitwise=False)
    assert gc_result == gc_expected


@pytest.mark.parametrize(
    "ham_terms,qwc_expected,gc_expected",
    [
        (
            0.9 * X(0) * Z(1) + 1.1 * X(0) + 0.8 * Z(0) + 0.5 * Z(0) * Z(1),
            [[X(0), X(0) * Z(1)], [Z(0), Z(0) * Z(1)]],
            [[X(0), X(0) * Z(1)], [Z(0), Z(0) * Z(1)]],
        ),
        (
            1.2 * X(0) * Y(1) * Z(2) + 1.1 * X(1) * X(2) + Z(1) * Y(2),
            [[X(0) * Y(1) * Z(2)], [X(1) * X(2)], [Z(1) * Y(2)]],
            [[X(0) * Y(1) * Z(2), X(1) * X(2), Z(1) * Y(2)]],
        ),
    ],
)
@pytest.mark.parametrize("method", ["graph", "sorted"])
def test_group_commuting_terms(ham_terms, qwc_expected, gc_expected, method):
    def canonical_group(group):
        """For sorting sympy Expr"""
        return tuple(sorted(srepr(expr) for expr in group))

    hamiltonian = SymbolicHamiltonian(ham_terms, nqubits=4)
    qwc_result = _group_commuting_terms(hamiltonian, qubitwise=True, method=method)
    assert sorted(map(canonical_group, qwc_result)) == sorted(map(canonical_group, qwc_expected))
    gc_result = _group_commuting_terms(hamiltonian, qubitwise=False, method=method)
    assert sorted(map(canonical_group, gc_result)) == sorted(map(canonical_group, gc_expected))


@pytest.mark.parametrize(
    "function_args,expected",
    [
        ({"symplectic_vector": np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8)}, ["X0", "Y1", "Z2"]),
        ({"symplectic_vector": np.array([0, 1, 0, 1, 0, 1, 0, 0], dtype=np.uint8)}, ["Y1", "X3"]),
    ],
)
def test_symplectic_to_pauli(function_args, expected):
    result = _symplectic_to_pauli(**function_args)
    assert result == expected


@pytest.mark.parametrize(
    "test,result",
    [
        (
            np.array(
                [[0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1]],
                dtype=np.uint8,
            ),
            np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1]], dtype=np.uint8),
        ),
        (
            np.array(
                [
                    [1, 1, 1, 1, 0, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0, 0, 1],
                    [1, 1, 1, 1, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0],
                ],
                dtype=np.uint8,
            ),
            np.array([[1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]], dtype=np.uint8),
        ),
        (
            np.array([[0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 1]], dtype=np.uint8),
            np.array([[0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]], dtype=np.uint8),
        ),
        (
            np.array([[0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0]], dtype=np.uint8),
            np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0]], dtype=np.uint8),
        ),
        (
            np.array(
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 1, 1],
                    [1, 1, 1, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 0, 0, 1],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
)
def test_binary_gaussian_elimination(test, result):
    # Hardcoded test results
    test = _binary_gaussian_elimination(test)
    assert np.allclose(test, result), f"RREF forms don't match: {test} != {result}"


def test_binary_nullspace():
    test_space = np.array(
        [[1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]], dtype=np.uint8
    )
    nullspace = _binary_nullspace(test_space)
    assert all(
        np.allclose((test_space @ vector) % 2, np.zeros(test_space.shape[0], dtype=np.uint8)) for vector in nullspace
    )


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
        dtype=np.uint8,
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
    nqubits = 6
    test_terms = (X(0) * X(2), Z(1) * X(3) * Z(4) * X(5), Z(0) * Z(2), Z(1), Z(3) * Z(5), Z(4))
    test_symplectic_form = np.array([_pauli_to_symplectic(term, nqubits=6) for term in test_terms])
    result_symplectic = _sort_tau_terms(test_symplectic_form)
    assert all(result_symplectic[i, i] or result_symplectic[i, i + nqubits] for i in range(nqubits))


def test_get_sigma_terms():
    test_terms = [X(0) * X(2), Z(1), Z(0) * Z(2), Z(1) * X(3) * Z(4) * X(5), Z(4), Z(3) * Z(5)]
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
                _pauli_to_symplectic(Z(0), 2),
            ],
            1,
        ),
        (
            [
                _pauli_to_symplectic(X(0) * X(1), 2),
            ],
            1,
        ),
        ([_pauli_to_symplectic(pauli, 2) for pauli in (X(0) * X(1), Y(0) * Y(1))], -1),
        ([_pauli_to_symplectic(pauli, 3) for pauli in (X(0), Y(1) * Z(2), Z(1) * Y(2))], 1),
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
        ],
        dtype=np.uint8,
    )
    phases = np.array([0, 0], dtype=np.uint8)
    gates_list = _col_reduce_x_matrix(stabiliser_matrix, phases)
    print(phases)
    # Single column operation, should have only CNOT gate
    assert len(gates_list) == 1 and gates_list[0].name == "cx"
    assert np.array_equal(phases, np.array([0, 0], dtype=np.uint8))
    # Code coverage for Gaussian elimination. Note: Input matrix isn't a commuting set, so shouldn't ever need
    control = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    stabiliser_matrix = np.array(control, dtype=np.uint8)
    phases = np.array([0, 0], dtype=np.uint8)
    _gates = _col_reduce_x_matrix(stabiliser_matrix, phases)
    # No change to stabiliser matrix
    assert np.array_equal(control, stabiliser_matrix)


def test_zero_z_matrix():
    stabiliser_matrix = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    phases = np.array([0, 0, 0, 0], dtype=np.uint8)
    gates_list = _zero_z_matrix(stabiliser_matrix, phases)
    # Single column operation, should have only CNOT gate
    assert len(gates_list) == 1 and gates_list[0].name == "s"
    assert phases[0] == 1
