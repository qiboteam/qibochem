"""
Tests for the UCC ansatz
"""

from functools import reduce

import numpy as np
import pytest
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import DepolarizingError, NoiseModel

from qibochem.ansatz import hf_circuit
from qibochem.ansatz.qeb import qeb_circuit
from qibochem.ansatz.ucc import expi_pauli, ucc_ansatz, ucc_circuit
from qibochem.ansatz.util import generate_excitations, mp2_amplitude, sort_excitations
from qibochem.driver import Molecule


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
    "excitation, mapping, basis_rotations, coeffs",
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
def test_ucc_circuit(excitation, mapping, basis_rotations, coeffs):
    """Build a UCC circuit with only one excitation"""
    theta = 0.1
    n_qubits = 4

    # Build the control array using SymbolicHamiltonian.circuit
    # But need to multiply theta by some coefficient introduced by the fermion->qubit mapping
    control_circuit = Circuit(n_qubits)
    for coeff, basis_rotation in zip(coeffs, basis_rotations):
        n_terms = len(basis_rotation)
        pauli_term = SymbolicHamiltonian(
            symbols.I(n_qubits - 1)
            * reduce(lambda x, y: x * y, (getattr(symbols, _op)(int(qubit)) for _op, qubit in basis_rotation.split()))
        )
        control_circuit += pauli_term.circuit(-coeff * theta)
    control_result = control_circuit(nshots=1)
    control_state = control_result.state(True)
    # Test the ucc_circuit function
    test_circuit = ucc_circuit(n_qubits, excitation, theta=theta, ferm_qubit_map=mapping)
    test_result = test_circuit(nshots=1)
    test_state = test_result.state(True)
    assert np.allclose(control_state, test_state)

    # Check that number of parametrised gates matches
    assert len(test_circuit.get_parameters()) == len(basis_rotations)


def test_ucc_circuit_noise_model():
    """Build a UCC circuit that uses all qubits"""
    excitation = [0, 1, 2, 3]
    mapping = None
    basis_rotations = (
                "X0 X1 Y2 X3",
                "Y0 Y1 Y2 X3",
                "Y0 X1 X2 X3",
                "X0 Y1 X2 X3",
                "Y0 X1 Y2 Y3",
                "X0 Y1 Y2 Y3",
                "X0 X1 X2 Y3",
                "Y0 Y1 X2 Y3",
            )
    coeffs = (-0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25)
    theta = 0.1
    n_qubits = 4

    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    # Build the control array using SymbolicHamiltonian.circuit
    # But need to multiply theta by some coefficient introduced by the fermion->qubit mapping
    control_circuit = Circuit(n_qubits)
    for coeff, basis_rotation in zip(coeffs, basis_rotations):
        n_terms = len(basis_rotation)
        pauli_term = SymbolicHamiltonian(
            symbols.I(n_qubits - 1)
            * reduce(lambda x, y: x * y, (getattr(symbols, _op)(int(qubit)) for _op, qubit in basis_rotation.split()))
        )
        control_circuit += pauli_term.circuit(-coeff * theta)
    control_circuit = noise_model.apply(control_circuit)
    for _i in range(control_circuit.nqubits):
        control_circuit.add(gates.M(_i))
    control_result = control_circuit(nshots=1000).frequencies()
    control_prob = {}
    for bitstring, count in control_result.items():
        control_prob[bitstring] = count/sum(control_result.values())
    # Test the ucc_circuit function
    test_circuit = ucc_circuit(n_qubits, excitation, theta=theta, ferm_qubit_map=mapping, noise_model=noise_model)
    for _i in range(test_circuit.nqubits):
        test_circuit.add(gates.M(_i))
    test_result = test_circuit(nshots=1000).frequencies()
    test_prob = {}
    for bitstring, count in test_result.items():
        test_prob[bitstring] = count/sum(test_result.values())
    # assert keys match
    assert control_prob.keys() == test_prob.keys()
    # assert values
    for key in control_prob:
        assert np.allclose(control_prob[key], test_prob[key], atol=1e-1)


@pytest.mark.parametrize(
    "excitation, mapping, basis_rotations, coeffs",
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
    ],
)
def test_qeb_circuit(excitation, mapping, basis_rotations, coeffs):
    """Build QEB circuit"""
    theta = 0.1
    n_qubits = 4

    # Build the control array using SymbolicHamiltonian.circuit
    # But need to multiply theta by some coefficient introduced by the fermion->qubit mapping
    control_circuit = Circuit(n_qubits)
    for coeff, basis_rotation in zip(coeffs, basis_rotations):
        n_terms = len(basis_rotation)
        pauli_term = SymbolicHamiltonian(
            symbols.I(n_qubits - 1)
            * reduce(lambda x, y: x * y, (getattr(symbols, _op)(int(qubit)) for _op, qubit in basis_rotation.split()))
        )
        control_circuit += pauli_term.circuit(-coeff * theta)
    control_result = control_circuit(nshots=1)
    control_state = control_result.state(True)

    test_circuit = qeb_circuit(n_qubits, excitation, theta=theta)
    test_result = test_circuit(nshots=1)
    test_state = test_result.state(True)
    assert np.allclose(control_state, test_state)


def test_qeb_circuit_noise_model():
    """Build QEB circuit"""
    excitation = [0, 1, 2, 3]
    mapping = None
    basis_rotations = (
                    "X0 X1 Y2 X3",
                    "Y0 Y1 Y2 X3",
                    "Y0 X1 X2 X3",
                    "X0 Y1 X2 X3",
                    "Y0 X1 Y2 Y3",
                    "X0 Y1 Y2 Y3",
                    "X0 X1 X2 Y3",
                    "Y0 Y1 X2 Y3",
                )
    coeffs = (-0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25)
    theta = 0.1
    n_qubits = 4

    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    # Build the control array using SymbolicHamiltonian.circuit
    # But need to multiply theta by some coefficient introduced by the fermion->qubit mapping
    control_circuit = Circuit(n_qubits)
    for coeff, basis_rotation in zip(coeffs, basis_rotations):
        n_terms = len(basis_rotation)
        pauli_term = SymbolicHamiltonian(
            symbols.I(n_qubits - 1)
            * reduce(lambda x, y: x * y, (getattr(symbols, _op)(int(qubit)) for _op, qubit in basis_rotation.split()))
        )
        control_circuit += pauli_term.circuit(-coeff * theta)
    control_circuit = noise_model.apply(control_circuit)
    for _i in range(control_circuit.nqubits):
        control_circuit.add(gates.M(_i))
    control_result = control_circuit(nshots=1000).frequencies()
    control_prob = {}
    for bitstring, count in control_result.items():
        control_prob[bitstring] = count/sum(control_result.values())
        
    # Test the ucc_circuit function
    test_circuit = qeb_circuit(n_qubits, excitation, theta=theta, noise_model=noise_model)
    for _i in range(test_circuit.nqubits):
        test_circuit.add(gates.M(_i))
    test_result = test_circuit(nshots=1000).frequencies()
    test_prob = {}
    for bitstring, count in test_result.items():
        test_prob[bitstring] = count/sum(test_result.values())
    # assert keys match
    assert control_prob.keys() == test_prob.keys()
    # assert values
    for key in control_prob:
        assert np.allclose(control_prob[key], test_prob[key], atol=1e-1)


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


def test_ucc_ansatz_h2_noise_model():
    """Test the default arguments of ucc_ansatz using H2"""
    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf()

    # Build control circuit
    control_circuit = hf_circuit(4, 2, noise_model=noise_model)
    excitations = ([0, 1, 2, 3], [0, 2], [1, 3])
    for excitation in excitations:
        control_circuit += ucc_circuit(4, excitation, noise_model=noise_model)

    test_circuit = ucc_ansatz(mol, noise_model=noise_model)

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


# def test_ucc_ansatz_embedding_noise_model():
#     """Test the default arguments of ucc_ansatz using LiH with HF embedding applied, but without the HF circuit"""
#     mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
#     mol.run_pyscf()
#     mol.hf_embedding(active=[1, 2, 5])

#     lam = 1.0
#     noise_model = NoiseModel()
#     noise_model.add(DepolarizingError(lam))

#     # Generate all possible excitations
#     excitations = []
#     for order in range(2, 0, -1):
#         # 2 electrons, 6 spin-orbitals
#         excitations += sort_excitations(generate_excitations(order, range(0, 2), range(2, 6)))
#     # Build control circuit
#     control_circuit = hf_circuit(6, 0, noise_model=noise_model)
#     for excitation in excitations:
#         control_circuit += ucc_circuit(6, excitation, noise_model=noise_model)
#     for _i in range(control_circuit.nqubits):
#         control_circuit.add(gates.M(_i))
#     control_result = control_circuit(nshots=1000).frequencies()
#     control_prob = {}
#     for bitstring, count in control_result.items():
#         control_prob[bitstring] = count/sum(control_result.values())

#     test_circuit = ucc_ansatz(mol, include_hf=False, use_mp2_guess=False, noise_model=noise_model)
#     for _i in range(test_circuit.nqubits):
#         test_circuit.add(gates.M(_i))
#     test_result = test_circuit(nshots=1000).frequencies()
#     test_prob = {}
#     for bitstring, count in test_result.items():
#         test_prob[bitstring] = count/sum(test_result.values())

#     # assert keys match
#     assert control_prob.keys() == test_prob.keys()
#     # assert values
#     for key in control_prob:
#         assert np.allclose(control_prob[key], test_prob[key], atol=1e-1)


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


# def test_ucc_ansatz_excitations_noise_model():
#     """Test the `excitations` argument of ucc_ansatz"""
#     mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
#     mol.run_pyscf()
#     mol.hf_embedding(active=[1, 2, 5])

#     lam = 1.0
#     noise_model = NoiseModel()
#     noise_model.add(DepolarizingError(lam))

#     # Generate all possible excitations
#     excitations = [[0, 1, 2, 3], [0, 1, 4, 5]]
#     # Build control circuit
#     control_circuit = hf_circuit(6, 2, noise_model=noise_model)
#     for excitation in excitations:
#         control_circuit += ucc_circuit(6, excitation, noise_model=noise_model)
#     for _i in range(control_circuit.nqubits):
#         control_circuit.add(gates.M(_i))

#     test_circuit = ucc_ansatz(mol, excitations=excitations, noise_model=noise_model)
#     for _i in range(test_circuit.nqubits):
#         test_circuit.add(gates.M(_i))

#     assert all(
#         control.name == test.name and control.target_qubits == test.target_qubits
#         for control, test in zip(list(control_circuit.queue), list(test_circuit.queue))
#     )
#     # Check that number of parametrised gates is the same
#     assert len(control_circuit.get_parameters()) == len(test_circuit.get_parameters())

#     control_result = control_circuit(nshots=1000).frequencies()
#     control_prob = {}
#     for bitstring, count in control_result.items():
#         control_prob[bitstring] = count/sum(control_result.values())

#     test_result = test_circuit(nshots=1000).frequencies()
#     test_prob = {}
#     for bitstring, count in test_result.items():
#         test_prob[bitstring] = count/sum(test_result.values())

#     # assert keys match
#     assert control_prob.keys() == test_prob.keys()
#     # assert values
#     for key in control_prob:
#         assert np.allclose(control_prob[key], test_prob[key], atol=1e-1)

def test_ucc_ansatz_error_checks():
    """Test the checks for input validity"""
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    # Define number of electrons and spin-obritals by hand
    mol.nelec = 4
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
