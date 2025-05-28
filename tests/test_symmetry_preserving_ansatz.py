"""
Tests for the symmetry preserving ansatz from Gard et al. (DOI: https://doi.org/10.1038/s41534-019-0240-1)
"""

import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.noise import DepolarizingError, NoiseModel

from qibochem.ansatz.symmetry import (
    a_gate,
    a_gate_indices,
    symm_preserving_circuit,
    x_gate_indices,
)


@pytest.mark.parametrize(
    "theta, phi, expected",
    [
        (None, None, np.array([0.0, 0.0, -1.0, 00])),
        (0.5 * np.pi, 0.0, np.array([0.0, 1.0, 0.0, 00])),
        (0.5 * np.pi, np.pi, np.array([0.0, -1.0, 0.0, 00])),
    ],
)
def test_a_gate(theta, phi, expected):
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    a_gates = a_gate(0, 1, theta=theta, phi=phi)
    circuit.add(a_gates)

    result = circuit(nshots=1)
    state_ket = result.state()
    assert np.allclose(state_ket, expected)


@pytest.mark.parametrize(
    "n_qubits, n_electrons, expected",
    [
        (4, 2, [0, 2]),
        (4, 3, [0, 1, 2]),
        (4, 4, [0, 1, 2, 3]),
    ],
)
def test_x_gate_indices(n_qubits, n_electrons, expected):
    test = x_gate_indices(n_qubits, n_electrons)
    assert test == expected


@pytest.mark.parametrize(
    "n_qubits, n_electrons, x_gates, expected",
    [
        (4, 2, [0, 2], 2 * [(0, 1), (2, 3), (1, 2)]),
        (6, 4, [0, 1, 2, 4], 3 * [(2, 3), (4, 5), (3, 4), (1, 2), (0, 1)]),
    ],
)
def test_a_gate_indices(n_qubits, n_electrons, x_gates, expected):
    test = a_gate_indices(n_qubits, n_electrons, x_gates)
    assert test == expected


@pytest.mark.parametrize(
    "n_qubits, n_electrons",
    [
        (4, 2),
        (6, 4),
    ],
)
def test_symm_preserving_circuit(n_qubits, n_electrons):
    control_circuit = Circuit(n_qubits)
    x_gates = x_gate_indices(n_qubits, n_electrons)
    control_circuit.add(gates.X(_i) for _i in x_gates)
    a_gate_qubits = a_gate_indices(n_qubits, n_electrons, x_gates)
    a_gates = [a_gate(qubit1, qubit2) for qubit1, qubit2 in a_gate_qubits]
    control_circuit.add(_gates for _a_gate in a_gates for _gates in _a_gate)

    test_circuit = symm_preserving_circuit(n_qubits, n_electrons)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )


@pytest.mark.parametrize(
    "n_qubits, n_electrons",
    [
        (4, 2),
        (6, 4),
    ],
)
def test_symm_preserving_circuit_noise_model(n_qubits, n_electrons):
    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    # control_circuit = Circuit(n_qubits)
    # x_gates = x_gate_indices(n_qubits, n_electrons)
    # control_circuit.add(gates.X(_i) for _i in x_gates)
    # a_gate_qubits = a_gate_indices(n_qubits, n_electrons, x_gates)
    # a_gates = [a_gate(qubit1, qubit2) for qubit1, qubit2 in a_gate_qubits]
    # control_circuit.add(_gates for _a_gate in a_gates for _gates in _a_gate)
    # control_circuit = noise_model.apply(control_circuit)

    test_circuit = symm_preserving_circuit(n_qubits, n_electrons, noise_model=noise_model)

    for _i in range(test_circuit.nqubits):
        test_circuit.add(gates.M(_i))
        # control_circuit.add(gates.M(_i))
        
    # counts = control_circuit(nshots=int(1e3)).frequencies()
    # probs = {}
    # for bitstring, count in counts.items():
    #     probs[bitstring] = count/sum(counts.values())
    test_counts = test_circuit(nshots=int(1e3)).frequencies()
    test_probs = {}
    for bitstring, count in test_counts.items():
        test_probs[bitstring] = count/sum(test_counts.values())

    ground_truth = {format(i, f'0{n_qubits}b'): 1/2**n_qubits for i in range(2**n_qubits)}
    # assert keys match
    assert ground_truth.keys() == test_probs.keys()
    # assert values
    for key in test_probs:
        assert np.allclose(ground_truth[key], test_probs[key], atol=1e-1)
