"""
Reworking the expectation_from_samples part of the expectation function
"""

from functools import reduce

import numpy as np

###
# Imports for the module
import qibo
from qibo import gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import I, X, Y, Z

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.driver.molecule import Molecule
from qibochem.measurement import expectation as expectation2

###


# from qibo.optimizers import optimize


# from qibo import gates


# Define molecule and populate
mol = Molecule(xyz_file="h2.xyz")
mol.run_pyscf()

# mol.hf_embedding(active=[1, 2, 5])


# hamiltonian = mol.hamiltonian(oei=mol.embed_oei, tei=mol.embed_tei, constant=mol.inactive_energy)
hamiltonian = mol.hamiltonian()

# Build circuit
circuit = hf_circuit(mol.nso, sum(mol.nelec))
# circuit = hf_circuit(mol.n_active_orbs, mol.n_active_e)

# Add random stuff to circuit:
# circuit.add(gates.RX(_i, 0.0) for _i in range(mol.nso))
# random_parameters = np.random.rand(len(circuit.get_parameters()))
# circuit.set_parameters(random_parameters)
# print(circuit.draw())


# for term in hamiltonian.terms:
#     _ham = SymbolicHamiltonian(I(0)*I(1)*I(2)*I(3)*term.coefficient*reduce(lambda x, y: x*y, term.factors, 1))
#     print(f"{term.factors}")
# print(len(hamiltonian.terms))
# coeffs = np.array([abs(term.coefficient.real) for term in hamiltonian.terms])


# quit()


def symbolic_to_sympy_term(symbolic_term):
    """Helper function to convert a Qibo SymbolicTerm to a SymPy term for re-building the SymbolicHamiltonian"""
    return symbolic_term.coefficient * reduce(lambda x, y: x * y, symbolic_term.factors, 1.0)


def split_xy_z_terms(hamiltonian):
    """
    Split up the Z and X/Y terms in a Hamiltonian

    Args:
        hamiltonian (qibo.hamiltonian.SymbolicHamiltonian): Molecular Hamiltonian

    Returns:
        z_ham, xy_ham: Two-tuple of SymbolicHamiltonians representing the Z and X/Y terms respectively
    """
    z_only_terms = [
        term for term in hamiltonian.terms if not any(factor.name[0] in ("X", "Y") for factor in term.factors)
    ]
    xy_terms = [term for term in hamiltonian.terms if term not in z_only_terms]
    # Convert the sorted SymbolicTerms back to SymbolicHamiltonians
    z_ham = SymbolicHamiltonian(sum(symbolic_to_sympy_term(term) for term in z_only_terms))
    xy_ham = SymbolicHamiltonian(sum(symbolic_to_sympy_term(term) for term in xy_terms))
    return z_ham, xy_ham


def old_pauli_term_sample_expectation(circuit, pauli_term, n_shots):
    """
    Calculate the expectation value of a single general Pauli string for a given circuit ansatz and number of shots

    Args:
        circuit (qibo.models.Circuit): Quantum circuit ansatz
        pauli_term (qibo.hamiltonians.terms.SymbolicTerm): General Pauli term
        n_shots (int): Number of times the circuit is run

    Returns:
        Expectation value of the Pauli term (float)
    """
    # Get the target qubits and basis rotation gates from the Hamiltonian term
    qubits = [int(factor.target_qubit) for factor in pauli_term.factors]
    basis = [type(factor.gate) for factor in pauli_term.factors]
    # Run a copy of the original circuit to get the output frequencies
    _circuit = circuit.copy()
    _circuit.add(gates.M(*qubits, basis=basis))
    result = _circuit(nshots=n_shots)
    frequencies = result.frequencies(binary=True)

    # Only works for Z terms, raises an error if ham_term has X/Y terms
    # single_term_ham = SymbolicHamiltonian(reduce(lambda x, y: x*y, pauli_term.factors, 1.0))
    # return single_term_ham.expectation_from_samples(frequencies, qubit_map=qubits)

    # Workaround code to handle X/Y terms in the Hamiltonian:
    # Get each Pauli string e.g. X0Y1
    pauli = [factor.name for factor in pauli_term.factors]
    # Replace each X and Y symbol with Z; then include the term coefficient
    pauli_z = [Z(int(element[1:])) for element in pauli]
    z_only_ham = SymbolicHamiltonian(pauli_term.coefficient * reduce(lambda x, y: x * y, pauli_z, 1.0))
    # Can now apply expectation_from_samples directly
    return z_only_ham.expectation_from_samples(frequencies, qubit_map=qubits)


def broadcast_pauli_term(pauli_term, n_qubits):
    """Add I s.t. pauli_term has a term for every qubit and return it as a list"""
    qubits_w_terms = {int(factor.target_qubit): factor.name[0] for factor in pauli_term.factors}
    return [getattr(qibo.symbols, qubits_w_terms[_i])(_i) if _i in qubits_w_terms else I(_i) for _i in range(n_qubits)]


def ordered_eigenvalues(pauli_term_list):
    """
    Returns the eigenvalues of pauli_term using kron

    Args:
        pauli_term (SymbolicTerm): A single Pauli term
    """
    individual_eigvals = [(1, -1) if factor.name[0] != "I" else (1, 1) for factor in pauli_term_list]
    return reduce(lambda x, y: np.kron(x, y), individual_eigvals)


def pauli_term_sample_expectation(circuit, pauli_term, n_shots):
    """
    Calculate the expectation value of a single general Pauli string for a given circuit ansatz and number of shots

    Args:
        circuit (qibo.models.Circuit): Quantum circuit ansatz
        pauli_term (qibo.hamiltonians.terms.SymbolicTerm): General Pauli term
        n_shots (int): Number of times the circuit is run

    Returns:
        Expectation value of the Pauli term (float)
    """
    # Fill in the Pauli term with I terms
    pauli_term_list = broadcast_pauli_term(pauli_term, circuit.nqubits)
    # print(pauli_term_list)
    # Get the target qubits and basis rotation gates from the Hamiltonian term
    measurement_gates = [
        # NotImplementedError: Basis rotation is not implemented for I
        gates.M(_i, basis=type(factor.gate)) if factor.name[0] != "I" else gates.M(_i, basis=gates.Z)
        for _i, factor in enumerate(pauli_term_list)
    ]
    # Run a copy of the original circuit to get the output probabilities
    _circuit = circuit.copy()
    _circuit.add(measurement_gates)
    result = _circuit(nshots=n_shots)
    probs = result.probabilities()

    return (pauli_term.coefficient * np.dot(ordered_eigenvalues(pauli_term_list), probs)).real


def allocate_shots(hamiltonian, n_shots=1000, method=None, max_shots_per_term=200, threshold=0.0001):
    """
    Allocate shots to each term in the Hamiltonian for the calculation of the expectation value

    Args:
        hamiltonian (SymbolicHamiltonian): Molecular Hamiltonian
        n_shots (int): Total number of shots to be allocated. Default: ``1000``
        method (str): How to allocate the shots. The available options are: ``c``/``coefficients``: ``n_shots`` is distributed based on the
            relative magnitudes of the term coefficients, ``u``/``uniform``: ``n_shots`` is distributed evenly amongst each term.
        max_shots_per_term (int): Upper limit for the number of shots allocated to an individual term
        threshold: Used in the ``coefficients`` method to decide which terms to ignore, i.e. no shots will be allocated to terms with
            coefficient < threshold*np.max(term_coefficients)

    Returns:
        shot_allocation: A list containing the number of shots to be used for each Pauli term respectively
    """
    if method is None:
        method = "coefficients"
    n_terms = len(hamiltonian.terms)

    shot_allocation = np.zeros(n_terms, dtype=int)
    while True:
        remaining_shots = n_shots - sum(shot_allocation)
        if not remaining_shots:
            break

        if method in ("c", "coefficients"):
            # Split shots based on the relative magnitudes of the coefficients of the Pauli terms
            # and only for terms that haven't reached the upper limit yet
            term_coefficients = np.array(
                [
                    abs(term.coefficient.real) if shots < max_shots_per_term else 0.0
                    for term, shots in zip(hamiltonian.terms, shot_allocation)
                ]
            )
            # Only keep terms that are at least threshold*largest_coefficient
            # Element-wise multiplication of the mask
            term_coefficients *= term_coefficients > threshold * np.max(term_coefficients)
            term_coefficients /= sum(term_coefficients)  # Normalise term_coefficients
            _shot_allocation = (remaining_shots * term_coefficients).astype(int)
            # Check that not too many shots have been allocated
            if sum(_shot_allocation) > remaining_shots:
                indices_to_deduct = np.nonzero(shot_allocation)[: (sum(_shot_allocation) - remaining_shots)]
                np.add.at(_shot_allocation, indices_to_deduct, -1)

            # Check to see if shots could be allocated?
            if not _shot_allocation.any():
                # No shots allocated, need to re-distribute the remaining shots
                # Check to see which terms have shots allocated, and haven't reached the max shot limit
                nonzero_indices = np.where((shot_allocation * (shot_allocation < max_shots_per_term)) != 0)[0]
                # Check if there are enough shots to allocate to the remaining terms equally
                if remaining_shots < len(nonzero_indices):
                    nonzero_indices = nonzero_indices[:remaining_shots]
                np.add.at(_shot_allocation, nonzero_indices, 1)
        elif method in ("u", "uniform"):
            # Even distribution of shots for every term
            _shot_allocation = np.array([remaining_shots // n_terms for _ in range(n_terms)])
            if not _shot_allocation.any():
                _shot_allocation = np.array(
                    [1 if (_i < remaining_shots and shots) else 0 for _i, shots in enumerate(shot_allocation)]
                )
        else:
            raise NameError("Unknown method!")

        # Add on to the current allocation, and set upper limit to the number of shots for a given term
        shot_allocation += _shot_allocation
        shot_allocation = np.clip(shot_allocation, 0, max_shots_per_term)

        # In case n_shots is too high s.t. max_shots_per_term is too low
        # Increase max_shots_per_term, and redo the shot allocation
        if np.min(shot_allocation) == max_shots_per_term:
            max_shots_per_term *= 2
            shot_allocation = np.zeros(n_terms, dtype=int)

    return shot_allocation.astype(int).tolist()


def expectation(
    circuit: qibo.models.Circuit,
    hamiltonian: SymbolicHamiltonian,
    from_samples: bool = False,
    n_shots: int = 1000,
    n_shots_per_pauli_term: bool = True,
    shot_allocation=None,
) -> float:
    """
    Calculate expectation value of some Hamiltonian using either the state vector or sample measurements from running a
    quantum circuit

    Args:
        circuit (qibo.models.Circuit): Quantum circuit ansatz
        hamiltonian (SymbolicHamiltonian): Molecular Hamiltonian
        from_samples (Boolean): Whether the expectation value calculation uses samples or the simulated
            state vector. Default: ``False``; Results are from a state vector simulation
        n_shots (int): Number of times the circuit is run if ``from_samples=True``. Default: ``1000``
        n_shots_per_pauli_term (Boolean): Whether or not ``n_shots`` is used for each Pauli term in the Hamiltonian, or for
            *all* the terms in the Hamiltonian. Default: ``True``; ``n_shots`` are used to get the expectation value for each
            term in the Hamiltonian.
        shot_allocation: Iterable containing the number of shots to be allocated to each term in the Hamiltonian respectively.
            Default: ``None``; then the ``allocate_shots`` function is called to build the list.

    Returns:
        Hamiltonian expectation value (float)
    """
    if from_samples:
        # n_shots is used to get the expectation value of each individual Pauli term
        if n_shots_per_pauli_term:
            # Split up the Z and non-Z terms
            z_ham, xy_ham = split_xy_z_terms(hamiltonian)
            # Then get the total expectation value
            # z_total = z_ham_sample_expectation(circuit, z_ham, n_shots)
            z_total = sum(pauli_term_sample_expectation(circuit, term, n_shots) for term in z_ham.terms)
            print(f"z_total: {z_total}")
            xy_total = sum(pauli_term_sample_expectation(circuit, term, n_shots) for term in xy_ham.terms)
            print(f"xy_total: {xy_total}")
            total = z_total + xy_total
        # Total of n_shots for ALL the terms in the Hamiltonian
        else:
            if shot_allocation is None:
                shot_allocation = allocate_shots(hamiltonian, n_shots)
            # print(shot_allocation)
            # Then sum up the individual Pauli terms in the Hamiltonian to get the overall expectation value
            total = sum(
                pauli_term_sample_expectation(circuit, term, shots)
                for shots, term in zip(shot_allocation, hamiltonian.terms)
            )
        # Add the constant term if present. Note: Energies (in chemistry) are all real values
        total += hamiltonian.constant.real
        return total


# expectation(
#     circuit: qibo.models.Circuit,
#     hamiltonian: SymbolicHamiltonian,
#     from_samples: bool=False,
#     n_shots: int=1000,
#     n_shots_per_pauli_term: bool=True,
#     shot_allocation=None
# )


# print(f"HF energy: {mol.e_hf:.9f} (Classical)")
print(f"Energy: {expectation2(circuit, hamiltonian):.9f} (Quantum)")

print(f"\nNew code!!!!")
print(f"nHamiltonian terms: {len(hamiltonian.terms)}")
n_shots = 2000

control = expectation(circuit, hamiltonian, from_samples=True, n_shots=n_shots)
print(f"Control: {control}")
print()

quit()

shot_allocation = allocate_shots(hamiltonian, n_shots=n_shots, method="u")
new_code1 = expectation(
    circuit,
    hamiltonian,
    from_samples=True,
    n_shots=n_shots,
    n_shots_per_pauli_term=False,
    shot_allocation=shot_allocation,
)
print(f"HF energy: {new_code1:.9f} (Quantum - Uniform allocation)")


shot_allocation = allocate_shots(hamiltonian, n_shots=n_shots, method="c", threshold=0.01)
new_code2 = expectation(
    circuit,
    hamiltonian,
    from_samples=True,
    n_shots=n_shots,
    n_shots_per_pauli_term=False,
    shot_allocation=shot_allocation,
)
print(f"HF energy: {new_code2:.9f} (Quantum - Coefficient allocation)")
print()
print()

new_code2 = expectation(circuit, hamiltonian, from_samples=True, n_shots=n_shots, n_shots_per_pauli_term=False)
print(f"HF energy: {new_code2:.9f} (Quantum - Coefficient allocation)")
# shot_allocation = [0 for _ in hamiltonian.terms]
# shot_allocation[0] = n_shots
# new_code2 = expectation(
#     circuit, hamiltonian, from_samples=True, n_shots=n_shots, n_shots_per_pauli_term=False, shot_allocation=shot_allocation
# )
# print(f"HF energy: {new_code2:.9f} (Quantum - Coefficient allocation)")
