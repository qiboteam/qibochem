"""
Tests for ansatz utility functions (util.py)
"""

from math import prod

import numpy as np
import pytest
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian

from qibochem.ansatz._ansatz import (
    _a_gate,
    _a_gate_indices,
    _basis_rotation_layout,
    _basis_rotation_unitary,
    _expi_pauli,
    _qr_decompose_givens,
    _x_gate_indices,
)
from qibochem.ansatz.ansatz import (
    basis_rotation_circuit,
    givens_circuit,
    he_circuit,
    hf_circuit,
    qeb_circuit,
    symm_preserving_circuit,
    ucc_circuit,
)
from qibochem.ansatz.utils import _sort_excitations, generate_excitations, mp2_amplitude
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


def test_basis_rotation_unitary():
    control = np.array(
        [
            [0.95041528, 0.0, -0.09834165, 0.0, -0.29502494, 0.0],
            [0.0, 0.9016556, 0.0, -0.19339968, 0.0, -0.38679937],
            [0.09834165, 0.0, 0.99504153, 0.0, -0.01487542, 0.0],
            [0.0, 0.19339968, 0.0, 0.98033112, 0.0, -0.03933776],
            [0.29502494, 0.0, -0.01487542, 0.0, 0.95537375, 0.0],
            [0.0, 0.38679937, 0.0, -0.03933776, 0.0, 0.92132448],
        ]
    )
    parameters = (-0.1, -0.2, -0.3, -0.4)
    unitary_matrix = _basis_rotation_unitary([0, 1], [2, 3, 4, 5], parameters=parameters)

    identity = np.eye(6)
    assert np.allclose(unitary_matrix @ unitary_matrix.T, identity)
    assert np.allclose(unitary_matrix, control)


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
    ref_z = [
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
    assert np.allclose(z_angles, ref_z)


@pytest.mark.parametrize(
    "nqubits,control",
    [
        (
            4,
            [
                (0, 0, -np.pi),
                (2, 0, 0.0),
                (1, 1, -0.5 * np.pi),
                (0, 2, -1.5207546393123066),
                (2, 2, -2.356194490192345),
                (1, 3, -0.5 * np.pi),
            ],
        ),
        (
            6,
            [
                (0, 0, -np.pi),
                (2, 0, 0.0),
                (4, 0, 0.0),
                (1, 1, -0.5 * np.pi),
                (3, 1, -0.5 * np.pi),
                (0, 2, -1.5207546393123066),
                (2, 2, -0.5 * np.pi),
                (4, 2, -2.356194490192345),
                (1, 3, -0.09995829685982476),
                (3, 3, -3.000171297352484),
                (0, 4, -1.5207546393123068),
                (2, 4, -0.5 * np.pi),
                (4, 4, -2.356194490192345),
                (1, 5, -0.5 * np.pi),
                (3, 5, -0.5 * np.pi),
            ],
        ),
    ],
)
def test_basis_rotation_layout(nqubits, control):
    z_angles = [
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
    test = _basis_rotation_layout(nqubits, z_angles)
    assert all(
        (test_item[0] == control_item[0]) and (test_item[1] == control_item[1])
        for test_item, control_item in zip(test, control)
    )
    assert all(np.isclose(test_item[2], control_item[2]) for test_item, control_item in zip(test, control))


@pytest.mark.parametrize(
    "parameters,include_hf,control_parameters",
    [
        (None, True, np.zeros(15)),
        (
            0.1,
            False,
            np.array(
                [
                    -np.pi,
                    0.0,
                    0.0,
                    -0.5 * np.pi,
                    -0.5 * np.pi,
                    -1.5207546393123066,
                    -0.5 * np.pi,
                    -2.356194490192345,
                    -0.09995829685982476,
                    -3.000171297352484,
                    -1.5207546393123068,
                    -0.5 * np.pi,
                    -2.356194490192345,
                    -0.5 * np.pi,
                    -0.5 * np.pi,
                ]
            ),
        ),
        (
            (-0.1, -0.2, -0.3, -0.4),
            False,
            np.array(
                [
                    0.0,
                    0.0,
                    -np.pi,
                    -0.5 * np.pi,
                    -0.5 * np.pi,
                    -1.5204181135485033,
                    -0.5 * np.pi,
                    -1.1071487177940904,
                    -2.8417187725525745,
                    -0.4472135954999578,
                    -1.62117454004129,
                    -0.5 * np.pi,
                    -2.0344439357957027,
                    -0.5 * np.pi,
                    -0.5 * np.pi,
                ]
            ),
        ),
    ],
)
def test_basis_rotation(parameters, include_hf, control_parameters):
    nqubits = 6
    nelectrons = 2

    # Generate the control circuit
    control_circuit = Circuit(nqubits)
    if include_hf:
        control_circuit.add(gates.X(_i) for _i in range(nelectrons))
    control_circuit.add(gates.GIVENS(_q + 1, _q, 0.0) for _ in range(3) for _q in (0, 2, 4, 1, 3))
    control_circuit.set_parameters(control_parameters)
    control_circuit.draw()

    test_circuit = basis_rotation_circuit(nqubits, nelectrons, parameters=parameters, include_hf=include_hf)
    test_circuit.draw()

    for gate, target in zip(control_circuit.queue, test_circuit.queue):
        assert gate.__class__.__name__ == target.__class__.__name__
        assert gate.qubits == target.qubits
        assert gate.target_qubits == target.target_qubits
        assert gate.control_qubits == target.control_qubits
        assert np.allclose(gate.parameters, target.parameters)


@pytest.mark.parametrize(
    "theta,phi,expected",
    [
        (0.5 * np.pi, 0.0, np.array([0.0, 1.0, 0.0, 00])),
        (0.5 * np.pi, np.pi, np.array([0.0, -1.0, 0.0, 00])),
    ],
)
def test_a_gate(theta, phi, expected):
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    a_gates = _a_gate(0, 1, theta=theta, phi=phi)
    circuit.add(a_gates)

    result = circuit(nshots=1)
    state_ket = result.state()
    assert np.allclose(state_ket, expected)


@pytest.mark.parametrize(
    "nqubits,nelectrons,expected",
    [
        (4, 2, [0, 2]),
        (4, 3, [0, 1, 2]),
        (4, 4, [0, 1, 2, 3]),
    ],
)
def test_x_gate_indices(nqubits, nelectrons, expected):
    test = _x_gate_indices(nqubits, nelectrons)
    assert test == expected


@pytest.mark.parametrize(
    "nqubits,nelectrons,x_gates,expected",
    [
        (4, 2, [0, 2], 2 * [(0, 1), (2, 3), (1, 2)]),
        (6, 4, [0, 1, 2, 4], 3 * [(2, 3), (4, 5), (3, 4), (1, 2), (0, 1)]),
    ],
)
def test_a_gate_indices(nqubits, nelectrons, x_gates, expected):
    test = _a_gate_indices(nqubits, nelectrons, x_gates)
    assert test == expected


@pytest.mark.parametrize(
    "nqubits,nelectrons,parameters,control_parameters",
    [
        (4, 2, [0.31415 for _ in range(24)], 6 * [-0.31415, -0.31415, 0.31415, 0.31415]),
        (4, 2, 0.1, 6 * [-0.1, -0.1, 0.1, 0.1]),
        (6, 4, None, [0.0 for _ in range(60)]),
    ],
)
def test_symm_preserving_circuit(nqubits, nelectrons, parameters, control_parameters):
    control_circuit = Circuit(nqubits)
    x_gates = _x_gate_indices(nqubits, nelectrons)
    control_circuit.add(gates.X(_i) for _i in x_gates)
    a_gate_qubits = _a_gate_indices(nqubits, nelectrons, x_gates)
    a_gates = [_a_gate(qubit1, qubit2, 0.2, 0.3) for qubit1, qubit2 in a_gate_qubits]
    control_circuit.add(_gates for _a_gate in a_gates for _gates in _a_gate)
    # Add 0.5*pi or pi to the control parameters
    n_a_gates = len(control_parameters) // 4
    control_parameters = np.array(control_parameters) + np.array(n_a_gates * [-np.pi, -0.5 * np.pi, 0.5 * np.pi, np.pi])
    control_circuit.set_parameters(control_parameters)

    test_circuit = symm_preserving_circuit(nqubits, nelectrons, parameters)

    for gate, target in zip(control_circuit.queue, test_circuit.queue):
        assert gate.__class__.__name__ == target.__class__.__name__
        assert gate.qubits == target.qubits
        assert gate.target_qubits == target.target_qubits
        assert gate.control_qubits == target.control_qubits
        assert np.allclose(gate.parameters, target.parameters)


def test_ansatz_argument_checks():
    """Input validity tests"""
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
    # basis_rotation_circuit: Wrong number of parameters
    too_few_params = [1]
    for circuit_func in (basis_rotation_circuit, symm_preserving_circuit):
        with pytest.raises(ValueError):
            _ = circuit_func(6, 2, parameters=too_few_params)


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
    "test,expected",
    [
        ([[0, 2], [0, 4], [1, 3], [1, 5]], [[0, 2], [1, 3], [0, 4], [1, 5]]),
        (
            [[0, 1, 2, 5], [0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5]],
            [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 5], [0, 1, 3, 4]],
        ),
    ],
)
def test_sort_excitations(test, expected):
    assert _sort_excitations(test) == expected


def test_sort_excitations_triples():
    with pytest.raises(NotImplementedError):
        _sort_excitations([[1, 2, 3, 4, 5, 6]])


def test_mp2_amplitude():
    # Single excitation
    assert mp2_amplitude([0, 2], np.random.rand(4), np.random.rand(4, 4)) == 0.0
    # Double excitation
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    l = mp2_amplitude([0, 1, 2, 3], h2.eps, h2.tei)
    ref_l = 0.06834019757197053
    assert np.isclose(l, ref_l)
