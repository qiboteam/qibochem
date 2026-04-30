"""
Test functionality to reduce the measurement cost of running VQE
"""

import numpy as np
import pytest
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from qibochem.measurement.optimization import (
    gc_measurement_mapping,
    measurement_basis_rotations,
)
from qibochem.measurement.shot_allocation import (
    allocate_shots,
    allocate_shots_by_variance,
)


def test_gc_measurement_mapping():
    """Remaining coverage tests for gc_measurement_mapping"""
    ham = SymbolicHamiltonian(Z(2))
    mapping, m_gates = gc_measurement_mapping(ham.form, 2, "izmaylov")
    assert mapping == {"Z2": ham.form}  # Single term expression should remain unchanged
    assert len(m_gates) == 1 and m_gates[0].name == "measure"  # Single measurement gate, no basis rotation

    ham = SymbolicHamiltonian(Z(0) + X(0))
    with pytest.raises(ValueError):
        _ = gc_measurement_mapping(ham.form, 2, "zc")


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
