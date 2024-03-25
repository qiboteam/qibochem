"""
Tests for the UCC ansatz and related functions
"""

from functools import reduce

import numpy as np
import pytest
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian

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
    "pauli_string",
    [
        "Z1",
        "Z0 Z1",
        "X0 X1",
        "Y0 Y1",
    ],
)
def test_expi_pauli(pauli_string):
    n_qubits = 2
    theta = 0.1

    # Build using exp(-i*theta*SymbolicHamiltonian)
    pauli_ops = sorted(((int(_op[1]), _op[0]) for _op in pauli_string.split()), key=lambda x: x[0])
    control_circuit = Circuit(n_qubits)
    pauli_term = SymbolicHamiltonian(
        symbols.I(n_qubits - 1)
        * reduce(lambda x, y: x * y, (getattr(symbols, pauli_op)(qubit) for qubit, pauli_op in pauli_ops))
    )
    control_circuit += pauli_term.circuit(-theta)
    control_result = control_circuit(nshots=1)
    control_state = control_result.state(True)

    test_circuit = expi_pauli(n_qubits, pauli_string, theta)
    test_result = test_circuit(nshots=1)
    test_state = test_result.state(True)

    assert np.allclose(control_state, test_state)


@pytest.mark.parametrize(
    "excitation,mapping,basis_rotations",
    [
        ([0, 2], None, ("Y0 X2", "X0 Y2")),  # JW singles
        # ([0, 2], None, ([("Y", 0), ("X", 2)], [("X", 0), ("Y", 2)])),  # JW singles
        # (
        #     [0, 1, 2, 3],
        #     None,
        #     (
        #         [("X", 0), ("X", 1), ("Y", 2), ("X", 3)],
        #         [("Y", 0), ("Y", 1), ("Y", 2), ("X", 3)],
        #         [("Y", 0), ("X", 1), ("X", 2), ("X", 3)],
        #         [("X", 0), ("Y", 1), ("X", 2), ("X", 3)],
        #         [("Y", 0), ("X", 1), ("Y", 2), ("Y", 3)],
        #         [("X", 0), ("Y", 1), ("Y", 2), ("Y", 3)],
        #         [("X", 0), ("X", 1), ("X", 2), ("Y", 3)],
        #         [("Y", 0), ("Y", 1), ("X", 2), ("Y", 3)],
        #     ),
        # ),  # JW doubles
        # ([0, 2], "bk", ([("X", 0), ("Y", 1), ("X", 2)], [("Y", 0), ("Y", 1), ("Y", 2)])),  # BK singles
    ],
)
def test_ucc_circuit(excitation, mapping, basis_rotations):
    """Build a UCC circuit with only one excitation"""
    theta = 0.1
    n_qubits = 4
    coeffs_dict = {2: (0.5, -0.5), 8: (-0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25)}

    coeffs = coeffs_dict[len(basis_rotations)]
    control_circuit = Circuit(n_qubits)
    for coeff, basis_rotation in zip(coeffs, basis_rotations):
        n_terms = len(basis_rotation)
        pauli_term = SymbolicHamiltonian(
            symbols.I(n_qubits - 1)
            * reduce(lambda x, y: x * y, (getattr(symbols, _op)(int(qubit)) for _op, qubit in basis_rotation.split()))
        )
        control_circuit += pauli_term.circuit(-coeff * theta)
    print(control_circuit.draw())
    print()
    control_result = control_circuit(nshots=1)
    control_state = control_result.state(True)

    test_circuit = ucc_circuit(n_qubits, excitation, theta=theta, ferm_qubit_map=mapping)
    test_result = test_circuit(nshots=1)
    test_state = test_result.state(True)

    print(test_circuit.draw())
    print()

    print(control_state)
    print()
    print(test_state)

    assert np.allclose(control_state, test_state)

    # Check that number of parametrised gates matches
    assert len(test_circuit.get_parameters()) == len(basis_rotations)


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
