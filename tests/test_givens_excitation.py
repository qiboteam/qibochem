"""
Tests for the symmetry preserving ansatz from Gard et al. (DOI: https://doi.org/10.1038/s41534-019-0240-1)
"""

import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.noise import DepolarizingError, NoiseModel

from qibochem.ansatz import hf_circuit
from qibochem.ansatz.givens_excitation import (
    double_excitation_gate,
    givens_excitation_ansatz,
    givens_excitation_circuit,
    single_excitation_gate,
)
from qibochem.ansatz.util import generate_excitations, mp2_amplitude, sort_excitations
from qibochem.driver import Molecule


def test_single_excitation_gate():
    # Hardcoded test
    theta = 0.1

    control_gates = [
        gates.CNOT(0, 1),
        gates.RY(0, 0.5 * theta),
        gates.CNOT(1, 0),
        gates.RY(0, -0.5 * theta),
        gates.CNOT(1, 0),
        gates.CNOT(0, 1),
    ]
    test_list = single_excitation_gate([0, 1, 2, 3], 0.1)

    # Check gates are correct
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(control_gates, test_list)
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
    test_list = double_excitation_gate([0, 1, 2, 3], 0.0)

    # Check gates are correct
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(control_gates, test_list)
    )


@pytest.mark.parametrize(
    "excitation, expected",
    [
        ([0, 2], single_excitation_gate([0, 2], 0.0)),
        ([0, 1, 2, 3], double_excitation_gate([0, 1, 2, 3], 0.0)),
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


@pytest.mark.parametrize(
    "excitation, expected",
    [
        ([0, 2], {"0000": 0.25, "0010": 0.25, "1000": 0.25, "1010": 0.25}),
        ([0, 1, 2, 3], {format(i, "04b"): 1 / 16 for i in range(16)}),
    ],
)
def test_givens_excitation_circuit_noise_model(excitation, expected):
    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))
    test_circuit = givens_excitation_circuit(4, excitation, noise_model=noise_model)
    for _i in range(test_circuit.nqubits):
        test_circuit.add(gates.M(_i))
    counts = test_circuit(nshots=int(1e3)).frequencies()
    probs = {}
    for bitstring, count in counts.items():
        probs[bitstring] = count / sum(counts.values())
    # assert keys match
    assert probs.keys() == expected.keys()
    # assert values
    for key in probs:
        assert np.allclose(probs[key], expected[key], atol=1e-1)


def test_givens_excitation_errors():
    """Input excitations are single or double?"""
    with pytest.raises(NotImplementedError):
        _test_circuit = givens_excitation_circuit(4, list(range(6)))


def test_givens_excitation_ansatz_h2():
    """Test the default arguments of ucc_ansatz using H2"""
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf()

    # Build control circuit
    control_circuit = hf_circuit(4, 2)
    excitations = ([0, 1, 2, 3], [0, 2], [1, 3])
    for excitation in excitations:
        control_circuit += givens_excitation_circuit(4, excitation)

    test_circuit = givens_excitation_ansatz(mol)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        # for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
        for control, test in zip(control_circuit.queue, test_circuit.queue)
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())

    # Then check that the circuit parameters are the MP2 guess parameters
    # Get the MP2 amplitudes first, then expand the list based on the excitation type
    mp2_guess_amplitudes = [mp2_amplitude([0, 1, 2, 3], mol.eps, mol.tei) for _ in range(8)]  # Doubles
    mp2_guess_amplitudes += [0.0, 0.0, 0.0, 0.0]  # Singles
    coeffs = np.array([-0.125, 0.125, -0.125, 0.125, 0.125, -0.125, 0.125, -0.125, 1.0, 1.0, 1.0, 1.0])
    mp2_guess_amplitudes = coeffs * np.array(mp2_guess_amplitudes)
    # Need to flatten the output of circuit.get_parameters() to compare it to mp2_guess_amplitudes
    test_parameters = np.array([_x for _tuple in test_circuit.get_parameters() for _x in _tuple])
    assert np.allclose(mp2_guess_amplitudes, test_parameters)


def test_givens_excitation_ansatz_h2_noise_model():
    """Test the measurement outcomes of ucc_ansatz with noise model using H2"""
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf()

    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    # Build control circuit
    control_circuit = hf_circuit(4, 2, noise_model=noise_model)
    excitations = ([0, 1, 2, 3], [0, 2], [1, 3])
    for excitation in excitations:
        control_circuit += givens_excitation_circuit(4, excitation, noise_model=noise_model)

    test_circuit = givens_excitation_ansatz(mol, noise_model=noise_model)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        # for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
        for control, test in zip(control_circuit.queue, test_circuit.queue)
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())

    for _i in range(control_circuit.nqubits):
        control_circuit.add(gates.M(_i))
        test_circuit.add(gates.M(_i))

    counts = control_circuit(nshots=int(1e3)).frequencies()
    probs = {}
    for bitstring, count in counts.items():
        probs[bitstring] = count / sum(counts.values())
    test_counts = test_circuit(nshots=int(1e3)).frequencies()
    test_probs = {}
    for bitstring, count in test_counts.items():
        test_probs[bitstring] = count / sum(test_counts.values())
    # assert keys match
    assert probs.keys() == test_probs.keys()
    # assert values
    for key in probs:
        assert np.allclose(probs[key], test_probs[key], atol=1e-1)


def test_givens_excitation_ansatz_embedding():
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
        control_circuit += givens_excitation_circuit(6, excitation)

    test_circuit = givens_excitation_ansatz(mol, include_hf=False, use_mp2_guess=False)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())

    # Check that the circuit parameters are all zeros
    test_parameters = np.array([_x for _tuple in test_circuit.get_parameters() for _x in _tuple])
    assert np.allclose(test_parameters, np.zeros(len(test_parameters)))


def test_givens_excitation_ansatz_embedding_noise_model():
    """Test the default arguments of ucc_ansatz using LiH with HF embedding applied, but without the HF circuit
    Also test measurement outcomes when noise model is applied."""
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    mol.run_pyscf()
    mol.hf_embedding(active=[1, 2, 5])

    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    # Generate all possible excitations
    excitations = []
    for order in range(2, 0, -1):
        # 2 electrons, 6 spin-orbitals
        excitations += sort_excitations(generate_excitations(order, range(0, 2), range(2, 6)))
    # Build control circuit
    control_circuit = hf_circuit(6, 0, noise_model=noise_model)
    for excitation in excitations:
        control_circuit += givens_excitation_circuit(6, excitation, noise_model=noise_model)

    test_circuit = givens_excitation_ansatz(mol, include_hf=False, use_mp2_guess=False, noise_model=noise_model)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())

    # Check that the circuit parameters are all zeros
    test_parameters = np.array([_x for _tuple in test_circuit.get_parameters() for _x in _tuple])
    assert np.allclose(test_parameters, np.zeros(len(test_parameters)))

    for _i in range(control_circuit.nqubits):
        control_circuit.add(gates.M(_i))
        test_circuit.add(gates.M(_i))

    counts = control_circuit(nshots=int(1e3)).frequencies()
    probs = {}
    for bitstring, count in counts.items():
        probs[bitstring] = count / sum(counts.values())
    test_counts = test_circuit(nshots=int(1e3)).frequencies()
    test_probs = {}
    for bitstring, count in test_counts.items():
        test_probs[bitstring] = count / sum(test_counts.values())
    # assert keys match
    assert probs.keys() == test_probs.keys()
    # assert values
    for key in probs:
        assert np.allclose(probs[key], test_probs[key], atol=1e-1)


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
        control_circuit += givens_excitation_circuit(6, excitation)

    test_circuit = givens_excitation_ansatz(mol, excitations=excitations)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())


def test_ucc_ansatz_excitations_noise_model():
    """Test the `excitations` argument of ucc_ansatz and measurement outcomes"""
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    mol.run_pyscf()
    mol.hf_embedding(active=[1, 2, 5])

    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    # Generate all possible excitations
    excitations = [[0, 1, 2, 3], [0, 1, 4, 5]]
    # Build control circuit
    control_circuit = hf_circuit(6, 2, noise_model=noise_model)
    for excitation in excitations:
        control_circuit += givens_excitation_circuit(6, excitation, noise_model=noise_model)

    test_circuit = givens_excitation_ansatz(mol, excitations=excitations, noise_model=noise_model)

    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
    )
    # Check that number of parametrised gates is the same
    assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())

    for _i in range(control_circuit.nqubits):
        control_circuit.add(gates.M(_i))
        test_circuit.add(gates.M(_i))

    counts = control_circuit(nshots=int(1e3)).frequencies()
    probs = {}
    for bitstring, count in counts.items():
        probs[bitstring] = count / sum(counts.values())
    test_counts = test_circuit(nshots=int(1e3)).frequencies()
    test_probs = {}
    for bitstring, count in test_counts.items():
        test_probs[bitstring] = count / sum(test_counts.values())
    # assert keys match
    assert probs.keys() == test_probs.keys()
    # assert values
    for key in probs:
        assert np.allclose(probs[key], test_probs[key], atol=1e-1)
