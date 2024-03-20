"""
Tests for the UCC ansatz and related functions
"""

from functools import partial

import numpy as np
import pytest
from qibo import gates

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.ucc import (
    expi_pauli,
    generate_excitations,
    mp2_amplitude,
    sort_excitations,
    ucc_ansatz,
    ucc_circuit,
)
from qibochem.driver import Molecule


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


@pytest.mark.parametrize(
    "pauli_string,qubit0_matrix",
    [
        ("Z1", np.identity(2)),
        ("Z0 Z1", np.diag((1, -1))),
    ],
)
def test_expi_pauli(pauli_string, qubit0_matrix):
    n_qubits = 2
    theta = 0.25 * np.pi
    dim = 2**n_qubits
    # Expected: cos(theta)*I + i*sin(theta)*(something \otimes Z)
    expected = np.cos(theta) * np.identity(dim) + 1.0j * np.sin(theta) * np.kron(qubit0_matrix, np.diag((1, -1)))
    # Test matrix
    circuit = expi_pauli(n_qubits, pauli_string, theta)
    test_matrix = np.identity(dim)
    for _gate in circuit.queue:
        if _gate.matrix().shape[0] != dim:
            _matrix = np.kron(np.identity(2), _gate.matrix())
        else:
            _matrix = _gate.matrix().T  # .T is to swap the control/target qubits
        test_matrix = np.dot(_matrix, test_matrix)
    assert np.allclose(np.diagonal(test_matrix), np.diagonal(expected))


@pytest.mark.parametrize(
    "excitation,mapping,basis_rotations",
    [
        ([0, 2], None, ([("Y", 0), ("X", 2)], [("X", 0), ("Y", 2)])),  # JW singles
        (
            [0, 1, 2, 3],
            None,
            (
                [("X", 0), ("X", 1), ("Y", 2), ("X", 3)],
                [("Y", 0), ("Y", 1), ("Y", 2), ("X", 3)],
                [("Y", 0), ("X", 1), ("X", 2), ("X", 3)],
                [("X", 0), ("Y", 1), ("X", 2), ("X", 3)],
                [("Y", 0), ("X", 1), ("Y", 2), ("Y", 3)],
                [("X", 0), ("Y", 1), ("Y", 2), ("Y", 3)],
                [("X", 0), ("X", 1), ("X", 2), ("Y", 3)],
                [("Y", 0), ("Y", 1), ("X", 2), ("Y", 3)],
            ),
        ),  # JW doubles
        ([0, 2], "bk", ([("X", 0), ("Y", 1), ("X", 2)], [("Y", 0), ("Y", 1), ("Y", 2)])),  # BK singles
    ],
)
def test_ucc_circuit(excitation, mapping, basis_rotations):
    """Build a UCC circuit with only one excitation"""
    gate_dict = {"X": gates.H, "Y": partial(gates.RX, theta=-0.5 * np.pi, trainable=False)}
    # Build the list of basis rotation gates
    basis_rotation_gates = [
        [gate_dict[_gate[0]](_gate[1]) for _gate in basis_rotation] for basis_rotation in basis_rotations
    ]
    # Build the CNOT cascade manually
    cnot_cascade = [gates.CNOT(_i, _i - 1) for _i in range(excitation[-1], excitation[0], -1)]
    cnot_cascade = cnot_cascade + [gates.RZ(excitation[0], 0.0)]
    cnot_cascade = cnot_cascade + [gates.CNOT(_i + 1, _i) for _i in range(excitation[0], excitation[-1])]

    # Build control list of gates
    nested_gate_list = [gate_list + cnot_cascade + gate_list for gate_list in basis_rotation_gates]
    gate_list = [_gate for gate_list in nested_gate_list for _gate in gate_list]

    # Test ucc_function
    circuit = ucc_circuit(4, excitation, ferm_qubit_map=mapping)
    # Check gates are correct
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(gate_list, list(circuit.queue))
    )
    # Check that only two parametrised gates
    assert len(circuit.get_parameters()) == len(basis_rotations)


def test_ucc_ferm_qubit_map_error():
    """If unknown fermion to qubit map used"""
    with pytest.raises(KeyError):
        ucc_circuit(2, [0, 1], ferm_qubit_map="Unknown")


def test_ucc_ansatz_h2():
    """Test the default arguments of ucc_ansatz using H2"""
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf()

    # Build control circuit
    control_circuit = hf_circuit(4, 2)
    excitations = ([0, 1, 2, 3], [0, 2], [1, 3])
    for excitation in excitations:
        control_circuit += ucc_circuit(4, excitation)

    test_circuit = ucc_ansatz(mol)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())

    # Then check that the circuit parameters are the MP2 guess parameters
    # Get the MP2 amplitudes first, then expand the list based on the excitation type
    mp2_guess_amplitudes = []
    for excitation in excitations:
        mp2_guess_amplitudes += [
            mp2_amplitude(excitation, mol.eps, mol.tei) for _ in range(2 ** (2 * (len(excitation) // 2) - 1))
        ]
    mp2_guess_amplitudes = np.array(mp2_guess_amplitudes)
    coeffs = np.array([-0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25, 0.0, 0.0, 0.0, 0.0])
    mp2_guess_amplitudes *= coeffs
    # Need to flatten the output of circuit.get_parameters() to compare it to mp2_guess_amplitudes
    test_parameters = np.array([_x for _tuple in test_circuit.get_parameters() for _x in _tuple])
    assert np.allclose(mp2_guess_amplitudes, test_parameters)


def test_ucc_ansatz_embedding():
    """Test the default arguments of ucc_ansatz using LiH with HF embedding applied, but without the HF circuit"""
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    mol.run_pyscf()
    mol.hf_embedding(active=[1, 2, 5])

    # Generate all possible excitations
    excitations = []
    for order in range(2, 0, -1):
        # 2 electrons, 6 spin-orbitals
        excitations += sort_excitations(generate_excitations(order, range(0, 2), range(2, 6)))
    # Build control circuit
    control_circuit = hf_circuit(6, 0)
    for excitation in excitations:
        control_circuit += ucc_circuit(6, excitation)

    test_circuit = ucc_ansatz(mol, include_hf=False, use_mp2_guess=False)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())

    # Check that the circuit parameters are all zeros
    test_parameters = np.array([_x for _tuple in test_circuit.get_parameters() for _x in _tuple])
    assert np.allclose(test_parameters, np.zeros(len(test_parameters)))


def test_ucc_ansatz_excitations():
    """Test the `excitations` argument of ucc_ansatz"""
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    mol.run_pyscf()
    mol.hf_embedding(active=[1, 2, 5])

    # Generate all possible excitations
    excitations = [[0, 1, 2, 3], [0, 1, 4, 5]]
    # Build control circuit
    control_circuit = hf_circuit(6, 2)
    for excitation in excitations:
        control_circuit += ucc_circuit(6, excitation)

    test_circuit = ucc_ansatz(mol, excitations=excitations)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())


def test_ucc_ansatz_error_checks():
    """Test the checks for input validity"""
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    # Define number of electrons and spin-obritals by hand
    mol.nelec = (2, 2)
    mol.nso = 12

    # Excitation level check: Input is in ("S", "D", "T", "Q")
    with pytest.raises(AssertionError):
        ucc_ansatz(mol, "Z")

    # Excitation level check: Excitation > doubles
    with pytest.raises(NotImplementedError):
        ucc_ansatz(mol, "T")

    # Excitations list check: excitation must have an even number of elements
    with pytest.raises(AssertionError):
        ucc_ansatz(mol, excitations=[[0]])

    # Input parameter check: Must have correct number of input parameters
    with pytest.raises(AssertionError):
        ucc_ansatz(mol, excitations=[[0, 1, 2, 3]], thetas=np.zeros(2))
