"""
Tests for ansatz utility functions (util.py)
"""

from math import prod

import numpy as np
import pytest
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian

from qibochem.ansatz._ansatz import _expi_pauli
from qibochem.ansatz.ansatz import he_circuit, hf_circuit, qeb_circuit, ucc_circuit
from qibochem.ansatz.utils import generate_excitations, mp2_amplitude, sort_excitations
from qibochem.driver import Molecule


@pytest.mark.parametrize(
    "rotation_gates,entangling_gate",
    [
        (["RX"], "CNOT"),
        ([gates.RZ], gates.CZ),
    ],
)
def test_he_circuit(rotation_gates, entangling_gate):
    """Test hardware efficient circuit"""
    nqubits = 4
    nlayers = 1
    control = Circuit(nqubits)
    for _ in range(nlayers):
        # Rotation gates
        control.add(
            (getattr(gates, rotation_gate) if isinstance(rotation_gate, str) else rotation_gate)(_i, 0.0)
            for rotation_gate in rotation_gates
            for _i in range(nqubits)
        )
        # Entanglement gates
        control.add(
            (getattr(gates, entangling_gate) if isinstance(entangling_gate, str) else entangling_gate)(_i, _i + 1)
            for _i in range(nqubits - 1)
        )
    # Test function
    test_circuit = he_circuit(nqubits, nlayers, rotation_gates, entangling_gate)

    for gate, target in zip(control.queue, test_circuit.queue):
        assert gate.__class__.__name__ == target.__class__.__name__
        assert gate.qubits == target.qubits
        assert gate.target_qubits == target.target_qubits
        assert gate.control_qubits == target.control_qubits
        assert gate.parameters == target.parameters


@pytest.mark.parametrize(
    "mapping,",
    [
        None,  # JW mapping
        "bk",  # BK mapping
    ],
)
def test_hf_circuit(mapping):
    """Tests the HF circuit for H2"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    hamiltonian = h2.hamiltonian(ferm_qubit_map=mapping)
    circuit = hf_circuit(h2.nso, h2.nelec, ferm_qubit_map=mapping)
    hf_energy = hamiltonian.expectation(circuit)

    # assert h2.e_hf == pytest.approx(hf_energy)
    assert pytest.approx(hf_energy) == h2_ref_energy


def test_mapping_errors():
    """Tests the HF circuit with an incorrect mapping"""
    with pytest.raises(KeyError):
        hf_circuit(4, 2, ferm_qubit_map="incorrect")


@pytest.mark.parametrize(
    "pauli_string",
    [
        "Z1",
        "Z0 Z1",
        "X0 X1",
        "Y0 Y1",
    ],
)
def test_expi_pauli(pauli_string):
    nqubits = 2
    theta = np.random.rand(1)

    # Build using exp(-i*theta*SymbolicHamiltonian)
    pauli_ops = sorted(((int(_op[1:]), _op[0]) for _op in pauli_string.split()), key=lambda x: x[0])
    control_circuit = Circuit(nqubits)
    pauli_term = SymbolicHamiltonian(
        symbols.I(nqubits - 1) * prod(getattr(symbols, pauli_op)(qubit) for qubit, pauli_op in pauli_ops)
    )
    control_circuit += pauli_term.circuit(-theta)
    control_result = control_circuit()
    control_state = control_result.state(True)

    test_circuit = _expi_pauli(nqubits, pauli_string, theta)
    test_result = test_circuit()
    test_state = test_result.state(True)

    assert np.allclose(control_state, test_state)


@pytest.mark.parametrize(
    "excitation,mapping,pauli_terms,coeffs",
    [
        ([0, 2], None, ("Y0 X2", "X0 Y2"), (0.5, -0.5)),  # JW singles
        (
            [0, 1, 2, 3],
            None,
            (
                "X0 X1 Y2 X3",
                "Y0 Y1 Y2 X3",
                "Y0 X1 X2 X3",
                "X0 Y1 X2 X3",
                "Y0 X1 Y2 Y3",
                "X0 Y1 Y2 Y3",
                "X0 X1 X2 Y3",
                "Y0 Y1 X2 Y3",
            ),
            (-0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25),
        ),  # JW doubles
        ([0, 2], "bk", ("X0 Y1 X2", "Y0 Y1 Y2"), (0.5, 0.5)),  # BK singles
    ],
)
def test_ucc_circuit(excitation, mapping, pauli_terms, coeffs):
    """Tests for ucc_circuit and qeb_circuit with only one excitation"""
    theta = 0.1
    nqubits = 4

    # Build the control array using SymbolicHamiltonian.circuit
    # But need to multiply theta by some coefficient introduced by the fermion->qubit mapping

    control_circuit = Circuit(nqubits)
    for coeff, pauli_string in zip(coeffs, pauli_terms):
        pauli_ops = sorted(((int(_op[1:]), _op[0]) for _op in pauli_string.split()), key=lambda x: x[0])
        pauli_term = SymbolicHamiltonian(
            symbols.I(nqubits - 1) * prod(getattr(symbols, pauli_op)(qubit) for qubit, pauli_op in pauli_ops)
        )
        control_circuit += pauli_term.circuit(-coeff * theta)
    control_result = control_circuit()
    control_state = control_result.state(True)
    # Test the ucc_circuit function
    if mapping is None:
        for circuit_func in (ucc_circuit, qeb_circuit):
            test_circuit = circuit_func(nqubits, excitation, theta=theta)
            test_result = test_circuit()
            test_state = test_result.state(True)
            assert np.allclose(control_state, test_state)
    else:
        test_circuit = ucc_circuit(nqubits, excitation, theta=theta, ferm_qubit_map=mapping)
        test_result = test_circuit()
        test_state = test_result.state(True)
        assert np.allclose(control_state, test_state)


def test_ucc_errors():
    """Validity of inputs to ucc_circuit and qeb_circuit"""
    # Excitation input errors
    for circuit_func in (ucc_circuit, qeb_circuit):
        for excitation in ([], [1000]):
            with pytest.raises(ValueError):
                circuit_func(2, excitation)
    # Other ucc_circuit checks
    with pytest.raises(NotImplementedError):  # Fermion to qubit mapping
        ucc_circuit(2, [0, 1], ferm_qubit_map="zc")
    with pytest.raises(ValueError):  # Trotter steps
        ucc_circuit(2, [0, 1], trotter_steps=0)


# Utility function tests
@pytest.mark.parametrize(
    "order,excite_from,excite_to,expected",
    [
        (1, [0, 1], [2, 3], [[0, 2], [1, 3]]),
        (2, [2, 3], [4, 5], [[2, 3, 4, 5]]),
        (3, [0, 1], [2, 3], [[]]),
    ],
)
def test_generate_excitations(order, excite_from, excite_to, expected):
    """Test generation of all possible excitations between two lists of orbitals"""
    test = generate_excitations(order, excite_from, excite_to)
    assert test == expected


@pytest.mark.parametrize(
    "order,excite_from,excite_to,expected",
    [
        (1, [0, 1], [2, 3, 4, 5], [[0, 2], [1, 3], [0, 4], [1, 5]]),
        (2, [0, 1], [2, 3, 4, 5], [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 5], [0, 1, 3, 4]]),
    ],
)
def test_sort_excitations(order, excite_from, excite_to, expected):
    test = sort_excitations(generate_excitations(order, excite_from, excite_to))
    assert test == expected


def test_sort_excitations_triples():
    with pytest.raises(NotImplementedError):
        sort_excitations([[1, 2, 3, 4, 5, 6]])


def test_mp2_amplitude_singles():
    assert mp2_amplitude([0, 2], np.random.rand(4), np.random.rand(4, 4)) == 0.0


def test_mp2_amplitude_doubles():
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    l = mp2_amplitude([0, 1, 2, 3], h2.eps, h2.tei)
    ref_l = 0.06834019757197053

    assert np.isclose(l, ref_l)
