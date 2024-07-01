import numpy as np
import pytest
from qibo import Circuit, gates

from qibochem.ansatz import symm_preserving_circuit
from qibochem.ansatz.symmetry import a_gate, a_gate_indices, x_gate_indices


@pytest.mark.parametrize(
    "theta,phi,expected",
    [
        (0.0, 0.0, np.array([0.0, 0.0, -1.0, 00])),
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


def test_x_gate_indices():
    # x_gate_indices(n_qubits, n_electrons)
    pass


def test_a_gate_indices():
    # a_gate_indices(n_qubits, n_electrons, x_gates
    pass


def test_symm_preserving_circuit():
    # symm_preserving_circuit(n_qubits, n_electrons
    pass


def test_vqe_symm_preserving_circuit():
    # Maybe don't do...?
    pass
