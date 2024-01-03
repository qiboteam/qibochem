from functools import partial

import numpy as np
import pytest
from qibo import gates
from scipy.optimize import minimize

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.ucc import (  # ucc_ansatz
    generate_excitations,
    mp2_amplitude,
    sort_excitations,
    ucc_ansatz,
    ucc_circuit,
)
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation


def test_generate_excitations_1():
    ex1 = generate_excitations(1, [0, 1], [2, 3])
    ref_ex1 = np.array([[0, 2], [1, 3]])

    assert np.allclose(ex1, ref_ex1)


def test_generate_excitations_2():
    ex2 = generate_excitations(2, [0, 1], [2, 3])
    ref_ex2 = np.array([[0, 1, 2, 3]])

    assert np.allclose(ex2, ref_ex2)


def test_generate_excitations_3():
    ex3 = generate_excitations(3, [0, 1], [2, 3])

    assert np.allclose(ex3, [[]])


def test_sort_excitations_1():
    ex1 = generate_excitations(1, [0, 1], [2, 3, 4, 5])
    sorted_excitations = sort_excitations(ex1)
    ref_sorted_ex1 = np.array([[0, 2], [1, 3], [0, 4], [1, 5]])

    assert np.allclose(sorted_excitations, ref_sorted_ex1)


def test_sort_excitations_2():
    ex2 = generate_excitations(2, [0, 1, 2, 3], [4, 5, 6, 7])
    sorted_excitations = sort_excitations(ex2)
    ref_sorted_ex2 = np.array(
        [
            [0, 1, 4, 5],
            [0, 1, 6, 7],
            [2, 3, 4, 5],
            [2, 3, 6, 7],
            [0, 1, 4, 7],
            [0, 1, 5, 6],
            [2, 3, 4, 7],
            [2, 3, 5, 6],
            [0, 3, 4, 5],
            [1, 2, 4, 5],
            [0, 3, 6, 7],
            [1, 2, 6, 7],
            [0, 2, 4, 6],
            [1, 3, 5, 7],
            [0, 3, 4, 7],
            [0, 3, 5, 6],
            [1, 2, 4, 7],
            [1, 2, 5, 6],
        ]
    )

    assert np.allclose(sorted_excitations, ref_sorted_ex2)


def test_sort_excitations_3():
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


def test_ucc_ansatz_h2():
    """Test the default arguments of ucc_ansatz using H2"""
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf()

    # Build control circuit
    control_circuit = hf_circuit(4, 2)
    for excitation in ([0, 1, 2, 3], [0, 2], [1, 3]):
        control_circuit += ucc_circuit(4, excitation)

    test_circuit = ucc_ansatz(mol)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())


def test_ucc_ansatz_h2():
    # Define molecule and populate
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.2))])
    mol.run_pyscf()

    # Apply embedding
    active_space = [1, 5]
    mol.hf_embedding(active=[1, 5])
    hamiltonian = mol.hamiltonian(oei=mol.embed_oei, tei=mol.embed_tei, constant=mol.inactive_energy)

    # Set parameters for the rest of the experiment
    n_qubits = mol.n_active_orbs
    n_electrons = mol.n_active_e

    excitations = generate_excitations(2, tuple(range(n_electrons)), tuple(range(n_electrons, n_qubits)))
    excitations += generate_excitations(1, tuple(range(n_electrons)), tuple(range(n_electrons, n_qubits)))
    n_excitations = len(excitations)

    # Build circuit
    circuit = hf_circuit(n_qubits, n_electrons)

    # UCCSD: Circuit
    all_coeffs = []
    for _ex in excitations:
        coeffs = []
        circuit += ucc_circuit(n_qubits, _ex, coeffs=coeffs)
        all_coeffs.append(coeffs)

    def electronic_energy(parameters):
        r"""
        Loss function for the UCCSD ansatz
        """
        all_parameters = []

        # UCC parameters
        # Expand the parameters to match the total UCC ansatz manually
        _ucc = parameters[:n_excitations]
        # Manually group the related excitations together
        ucc_parameters = [_ucc[0], _ucc[1], _ucc[2]]
        # Need to iterate through each excitation this time
        for _coeffs, _parameter in zip(all_coeffs, ucc_parameters):
            # Convert a single value to a array with dimension=n_param_gates
            ucc_parameter = np.repeat(_parameter, len(_coeffs))
            # Multiply by coeffs
            ucc_parameter *= _coeffs
            all_parameters.append(ucc_parameter)

        # Flatten all_parameters into a single list to set the circuit parameters
        all_parameters = [_x for _param in all_parameters for _x in _param]
        circuit.set_parameters(all_parameters)

        return expectation(circuit, hamiltonian)

    # Random initialization
    params = np.zeros(n_excitations)
    vqe = minimize(electronic_energy, params)

    lih_uccsd_energy = -7.847535097575567

    assert vqe.fun == pytest.approx(lih_uccsd_energy)
