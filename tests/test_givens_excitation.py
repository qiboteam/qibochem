"""
Tests for the symmetry preserving ansatz from Gard et al. (DOI: https://doi.org/10.1038/s41534-019-0240-1)
"""

import numpy as np
import pytest
from qibo import Circuit, gates

from qibochem.ansatz.givens_excitation import (
    double_excitation_gate,
    givens_excitation_ansatz,
    givens_excitation_circuit,
)


def test_double_excitation_gate():
    # Hardcoded test
    theta = 0.0

    control_gates = [
        gates.CNOT(2, 3),
        gates.CNOT(0, 2),
        gates.H(0),
        gates.H(3),
        gates.CNOT(0, 1),
        gates.CNOT(2, 3),
        gates.RY(0, -theta),
        gates.RY(1, theta),
        gates.CNOT(0, 3),
        gates.H(3),
        gates.CNOT(3, 1),
        gates.RY(0, -theta),
        gates.RY(1, theta),
        gates.CNOT(2, 1),
        gates.CNOT(2, 0),
        gates.RY(0, theta),
        gates.RY(1, -theta),
        gates.CNOT(3, 1),
        gates.H(3),
        gates.CNOT(0, 3),
        gates.RY(0, theta),
        gates.RY(1, -theta),
        gates.CNOT(0, 1),
        gates.CNOT(2, 0),
        gates.H(0),
        gates.H(3),
        gates.CNOT(0, 2),
        gates.CNOT(2, 3),
    ]
    test_list = double_excitation_gate([0, 1, 2, 3])

    for gate in test_list:
        if gate.parameters:
            print(gate.name, gate.parameters)

    # Check gates are correct
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(control_gates, test_list)
    )


@pytest.mark.parametrize(
    "excitation,expected",
    [
        ([0, 2], [gates.GIVENS(0, 2, 0.0)]),
        ([0, 1, 2, 3], double_excitation_gate([0, 1, 2, 3])),
    ],
)
def test_givens_excitation_circuit(excitation, expected):
    test_circuit = givens_excitation_circuit(4, excitation)
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(expected, list(test_circuit.queue))
    )
    # Check parameters of parametrized gates are all 0.0
    assert all(np.isclose(gate.parameters[0], 0.0) for gate in test_circuit.queue if gate.parameters)


def test_givens_excitation_errors():
    """Input excitations are single or double?"""
    with pytest.raises(NotImplementedError):
        test_circuit = givens_excitation_circuit(4, list(range(6)))
