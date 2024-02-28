from functools import partial

import numpy as np
import pytest
from qibo import gates
from scipy.optimize import minimize

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.ucc import (
    generate_excitations,
    mp2_amplitude,
    sort_excitations,
    ucc_ansatz,
    ucc_circuit,
)
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation


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


def test_ucc_jw_singles():
    """Build a UCC singles (JW) circuit"""
    rx_gate = partial(gates.RX, theta=-0.5 * np.pi, trainable=False)
    cnot_cascade = [gates.CNOT(2, 1), gates.CNOT(1, 0), gates.RZ(0, 0.0), gates.CNOT(1, 0), gates.CNOT(2, 1)]
    basis_rotation_gates = ([rx_gate(0), gates.H(2)], [gates.H(0), rx_gate(2)])

    # Build control list of gates
    nested_gate_list = [gate_list + cnot_cascade + gate_list for gate_list in basis_rotation_gates]
    gate_list = [_gate for gate_list in nested_gate_list for _gate in gate_list]

    # Test ucc_function
    circuit = ucc_circuit(4, [0, 2], ferm_qubit_map="jw")
    # Check gates are correct
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(gate_list, list(circuit.queue))
    )
    # Check that only two parametrised gates
    assert len(circuit.get_parameters()) == 2


def test_ucc_jw_doubles():
    """Build a UCC doubles (JW) circuit"""
    rx_gate = partial(gates.RX, theta=-0.5 * np.pi, trainable=False)
    cnot_cascade = [
        gates.CNOT(3, 2),
        gates.CNOT(2, 1),
        gates.CNOT(1, 0),
        gates.RZ(0, 0.0),
        gates.CNOT(1, 0),
        gates.CNOT(2, 1),
        gates.CNOT(3, 2),
    ]
    basis_rotation_gates = (
        [gates.H(0), gates.H(1), rx_gate(2), gates.H(3)],
        [rx_gate(0), rx_gate(1), rx_gate(2), gates.H(3)],
        [rx_gate(0), gates.H(1), gates.H(2), gates.H(3)],
        [gates.H(0), rx_gate(1), gates.H(2), gates.H(3)],
        [rx_gate(0), gates.H(1), rx_gate(2), rx_gate(3)],
        [gates.H(0), rx_gate(1), rx_gate(2), rx_gate(3)],
        [gates.H(0), gates.H(1), gates.H(2), rx_gate(3)],
        [rx_gate(0), rx_gate(1), gates.H(2), rx_gate(3)],
    )

    # Build control list of gates
    nested_gate_list = [gate_list + cnot_cascade + gate_list for gate_list in basis_rotation_gates]
    gate_list = [_gate for gate_list in nested_gate_list for _gate in gate_list]

    # Test ucc_function
    circuit = ucc_circuit(4, [0, 1, 2, 3], ferm_qubit_map="jw")
    # Check gates are correct
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(gate_list, list(circuit.queue))
    )
    # Check that only two parametrised gates
    assert len(circuit.get_parameters()) == 8


def test_ucc_bk_singles():
    """Build a UCC doubles (BK) circuit"""
    rx_gate = partial(gates.RX, theta=-0.5 * np.pi, trainable=False)
    cnot_cascade = [gates.CNOT(2, 1), gates.CNOT(1, 0), gates.RZ(0, 0.0), gates.CNOT(1, 0), gates.CNOT(2, 1)]
    basis_rotation_gates = ([gates.H(0), rx_gate(1), gates.H(2)], [rx_gate(0), rx_gate(1), rx_gate(2)])

    # Build control list of gates
    nested_gate_list = [gate_list + cnot_cascade + gate_list for gate_list in basis_rotation_gates]
    gate_list = [_gate for gate_list in nested_gate_list for _gate in gate_list]

    # Test ucc_function
    circuit = ucc_circuit(4, [0, 2], ferm_qubit_map="bk")
    # Check gates are correct
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(gate_list, list(circuit.queue))
    )
    # Check that only two parametrised gates
    assert len(circuit.get_parameters()) == 2


def test_ucc_ferm_qubit_map_error():
    """If unknown fermion to qubit map used"""
    with pytest.raises(KeyError):
        ucc_circuit(2, [0, 1], ferm_qubit_map="Unknown")


def test_ucc_parameter_coefficients():
    """Coefficients used to multiply the parameters in the UCC circuit. Note: may change in future!"""
    # UCC-JW singles
    control_values = (-1.0, 1.0)
    coeffs = []
    circuit = ucc_circuit(2, [0, 1], coeffs=coeffs)
    # Check that the signs of the coefficients have been saved
    assert all(control == test for control, test in zip(control_values, coeffs))

    # UCC-JW doubles
    control_values = (-0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25)
    coeffs = []
    circuit = ucc_circuit(4, [0, 1, 2, 3], coeffs=coeffs)
    # Check that the signs of the coefficients have been saved
    assert all(control == test for control, test in zip(control_values, coeffs))


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
