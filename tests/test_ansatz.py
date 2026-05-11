"""
Tests for ansatz utility functions (util.py)
"""

from math import prod

import numpy as np
import pytest
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian

from qibochem.ansatz._ansatz import (
    _basis_rotation_layout,
    _basis_rotation_unitary,
    _expi_pauli,
    _qr_decompose_givens,
)
from qibochem.ansatz.ansatz import (
    givens_circuit,
    he_circuit,
    hf_circuit,
    qeb_circuit,
    ucc_circuit,
)
from qibochem.ansatz.utils import generate_excitations, mp2_amplitude, sort_excitations
from qibochem.driver import Molecule


@pytest.mark.parametrize(
    "rotation_gates,entangling_gate",
    [
        (None, gates.CNOT),
        (["RX"], "CNOT"),
        ([gates.RZ], gates.CZ),
    ],
)
def test_he_circuit(rotation_gates, entangling_gate):
    """Test hardware efficient circuit"""
    nqubits = 4
    nlayers = 1
    # Test circuit first; convert the None arguments for the control circuit later
    test_circuit = he_circuit(nqubits, nlayers, rotation_gates, entangling_gate)

    control_circuit = Circuit(nqubits)
    rotation_gates = rotation_gates if rotation_gates is not None else ["RY", gates.RZ]
    for _ in range(nlayers):
        # Rotation gates
        control_circuit.add(
            (getattr(gates, rotation_gate) if isinstance(rotation_gate, str) else rotation_gate)(_i, 0.0)
            for _i in range(nqubits)
            for rotation_gate in rotation_gates
        )
        # Entanglement gates
        control_circuit.add(
            (getattr(gates, entangling_gate) if isinstance(entangling_gate, str) else entangling_gate)(_i, _i + 1)
            for _i in range(nqubits - 1)
        )

    for gate, target in zip(control_circuit.queue, test_circuit.queue):
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


def _givens_single_excitation(sorted_orbitals, theta):
    """
    Testing helper function: Decomposition of a Givens single excitation gate into single qubit rotations and CNOTs

    Args:
        sorted_orbitals (Sequence[int]): Sorted list of orbitals involved in the excitation
        theta (float): Rotation angle

    Returns:
        (list[Gate]): Decomposition of the Givens' single excitation gate
    """
    result = []
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[1]))
    result.append(gates.RY(sorted_orbitals[0], 0.5 * theta))
    result.append(gates.CNOT(sorted_orbitals[1], sorted_orbitals[0]))
    result.append(gates.RY(sorted_orbitals[0], -0.5 * theta))
    result.append(gates.CNOT(sorted_orbitals[1], sorted_orbitals[0]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[1]))
    return result


def _givens_double_excitation(sorted_orbitals, theta):
    """
    Testing helper function: Decomposition of a Givens double excitation gate into single qubit rotations and CNOTs

    Args:
        sorted_orbitals (Sequence[int]): Sorted list of orbitals involved in the excitation
        theta (float): Rotation angle

    Returns:
        (list[Gate]): Decomposition of the Givens' double excitation gate
    """
    result = []
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[2]))
    result.append(gates.H(sorted_orbitals[0]))
    result.append(gates.H(sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[1]))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[3]))
    result.append(gates.RY(sorted_orbitals[0], -0.125 * theta))
    result.append(gates.RY(sorted_orbitals[1], 0.125 * theta))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[3]))
    result.append(gates.H(sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[3], sorted_orbitals[1]))
    result.append(gates.RY(sorted_orbitals[0], -0.125 * theta))
    result.append(gates.RY(sorted_orbitals[1], 0.125 * theta))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[1]))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[0]))
    result.append(gates.RY(sorted_orbitals[0], 0.125 * theta))
    result.append(gates.RY(sorted_orbitals[1], -0.125 * theta))
    result.append(gates.CNOT(sorted_orbitals[3], sorted_orbitals[1]))
    result.append(gates.H(sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[3]))
    result.append(gates.RY(sorted_orbitals[0], 0.125 * theta))
    result.append(gates.RY(sorted_orbitals[1], -0.125 * theta))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[1]))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[0]))
    result.append(gates.H(sorted_orbitals[0]))
    result.append(gates.H(sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[2]))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[3]))
    return result


def test_givens_circuit():
    """Test givens_circuit using circuit decomposition given in reference paper"""
    nqubits = 4
    theta = 0.27183
    for excitation, decomposition in zip(
        ([0, 2], [0, 1, 2, 3]), (_givens_single_excitation, _givens_double_excitation)
    ):
        control_circuit = Circuit(nqubits)
        control_circuit.add(decomposition(excitation, theta))
        control_result = control_circuit()
        control_state = control_result.state(True)

        test_circuit = givens_circuit(nqubits, excitation, theta)
        test_result = test_circuit()
        test_state = test_result.state(True)

        assert np.allclose(control_state, test_state)


@pytest.mark.parametrize(
    "parameters,test",
    [
        (
            0.1,
            np.array(
                [
                    [0.99001666, 0.0, 0.099667, 0.0, 0.099667, 0.0],
                    [0.0, 0.99001666, 0.0, 0.099667, 0.0, 0.099667],
                    [-0.099667, 0.0, 0.99500833, 0.0, -0.00499167, 0.0],
                    [0.0, -0.099667, 0.0, 0.99500833, 0.0, -0.00499167],
                    [-0.099667, 0.0, -0.00499167, 0.0, 0.99500833, 0.0],
                    [0.0, -0.099667, 0.0, -0.00499167, 0.0, 0.99500833],
                ]
            ),
        ),
        (
            (-0.1, -0.2, -0.3, -0.4),
            np.array(
                [
                    [0.95041528, 0.0, -0.09834165, 0.0, -0.29502494, 0.0],
                    [0.0, 0.9016556, 0.0, -0.19339968, 0.0, -0.38679937],
                    [0.09834165, 0.0, 0.99504153, 0.0, -0.01487542, 0.0],
                    [0.0, 0.19339968, 0.0, 0.98033112, 0.0, -0.03933776],
                    [0.29502494, 0.0, -0.01487542, 0.0, 0.95537375, 0.0],
                    [0.0, 0.38679937, 0.0, -0.03933776, 0.0, 0.92132448],
                ]
            ),
        ),
    ],
)
def test_basis_rotation_unitary(parameters, test):
    occupied = range(0, 2)
    virtual = range(2, 6)

    unitary_matrix, _parameters = _basis_rotation_unitary(occupied, virtual, parameters=parameters)

    identity = np.eye(6)
    assert np.allclose(unitary_matrix @ unitary_matrix.T, identity)
    assert np.allclose(unitary_matrix, test)

    too_many_params = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    with pytest.raises(IndexError):
        _, _ = _basis_rotation_unitary(occupied, virtual, parameters=too_many_params)


def test_qr_decompose_givens():
    # Test case from test_basis_rotation_unitary
    unitary_matrix = np.array(
        [
            [0.99001666, 0.0, 0.099667, 0.0, 0.099667, 0.0],
            [0.0, 0.99001666, 0.0, 0.099667, 0.0, 0.099667],
            [-0.099667, 0.0, 0.99500833, 0.0, -0.00499167, 0.0],
            [0.0, -0.099667, 0.0, 0.99500833, 0.0, -0.00499167],
            [-0.099667, 0.0, -0.00499167, 0.0, 0.99500833, 0.0],
            [0.0, -0.099667, 0.0, -0.00499167, 0.0, 0.99500833],
        ]
    )
    z_angles = _qr_decompose_givens(unitary_matrix)
    ref_z = np.array(
        [
            -np.pi,
            -0.5 * np.pi,
            -2.356194490192345,
            0.0,
            -0.5 * np.pi,
            -1.5207546393123066,
            -0.5 * np.pi,
            -0.5 * np.pi,
            -3.000171297352484,
            -2.356194490192345,
            0.0,
            -0.5 * np.pi,
            -0.5 * np.pi,
            -0.09995829685982476,
            -1.5207546393123068,
        ]
    )

    assert np.allclose(z_angles, ref_z)


def test_basis_rotation_layout():
    control = np.array(
        [
            [0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [1, 0, 6, 0, 15, 0, 28, 0, 45, 0],
            [0, 5, 0, 14, 0, 27, 0, 44, 0, 29],
            [4, 0, 13, 0, 26, 0, 43, 0, 30, 0],
            [0, 12, 0, 25, 0, 42, 0, 31, 0, 16],
            [11, 0, 24, 0, 41, 0, 32, 0, 17, 0],
            [0, 23, 0, 40, 0, 33, 0, 18, 0, 7],
            [22, 0, 39, 0, 34, 0, 19, 0, 8, 0],
            [0, 38, 0, 35, 0, 20, 0, 9, 0, 2],
            [37, -1, 36, -1, 21, -1, 10, -1, 3, -1],
        ]
    )
    assert np.allclose(_basis_rotation_layout(10), control)


def test_ansatz_argument_checks():
    """Input validity of hf_circuit, ucc_circuit, qeb_circuit, and givens_circuit"""
    # Fermion to qubit mapping checks
    for circuit_func in (hf_circuit, ucc_circuit):
        with pytest.raises(NotImplementedError):
            circuit_func(2, [0, 1], ferm_qubit_map="zc")
    # Excitation input errors
    for circuit_func in (ucc_circuit, qeb_circuit, givens_circuit):
        for excitation in ([], [1000]):
            with pytest.raises(ValueError):
                circuit_func(2, excitation)
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
