from functools import reduce

import numpy as np
import qibo
from qibo import gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z


def pauli_term_sample_expectation(circuit, pauli_term, n_shots):
    """
    Calculate the expectation value of a general Pauli string for a given circuit ansatz and number of shots

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
    # single_term_ham = SymbolicHamiltonian(reduce(lambda x, y: x*y, pauli_term.factors, 1))
    # return single_term_ham.expectation_from_samples(frequencies, qubit_map=qubits)

    # Workaround code to handle X/Y terms in the Hamiltonian:
    # Get each Pauli string e.g. X0Y1
    pauli = [factor.name for factor in pauli_term.factors]
    # Replace each X and Y symbol with Z; then include the term coefficient
    pauli_z = [Z(int(element[1:])) for element in pauli]
    z_only_ham = SymbolicHamiltonian(pauli_term.coefficient * reduce(lambda x, y: x * y, pauli_z, 1.0))
    # Can now apply expectation_from_samples directly
    return z_only_ham.expectation_from_samples(frequencies, qubit_map=qubits)


def allocate_shots(hamiltonian, n_shots=1000, method=None, threshold=0.05):
    """
    Allocate shots to each term in the Hamiltonian for the calculation of the expectation value

    Args:
        hamiltonian (SymbolicHamiltonian): Molecular Hamiltonian
        n_shots (int): Total number of shots to be allocated. Default: ``1000``
        method (str): How to allocate the shots. The available options are: ``c``/``coefficients``: ``n_shots`` is distributed
            based on the relative magnitudes of the term coefficients, ``u``/``uniform``: ``n_shots`` is distributed evenly
            amongst each term.
        threshold: Used in the ``coefficients`` method to decide which terms to ignore, i.e. no shots will be allocated to
            terms with coefficient < threshold*np.max(term_coefficients)

    Returns:
        shot_allocation: A list containing the number of shots to be used for each Pauli term respectively
    """
    if method is None:
        method = "c"
    n_terms = len(hamiltonian.terms)
    shot_allocation = []

    if method in ("c", "coefficients"):
        # Split shots based on the relative magnitudes of the coefficients of the Pauli terms
        term_coefficients = np.array([abs(term.coefficient.real) for term in hamiltonian.terms])
        # Only keep terms that are at least threshold*largest_coefficient
        # Element-wise multiplication of the mask
        term_coefficients *= term_coefficients > threshold * np.max(term_coefficients)
        term_coefficients /= sum(term_coefficients)  # Normalise term_coefficients
        shot_allocation = (n_shots * term_coefficients).astype(int)
    elif method in ("u", "uniform"):
        # Split evenly amongst all the terms in the Hamiltonian
        shot_allocation = np.array([n_shots // n_terms for _ in range(n_terms)])
    else:
        raise NameError("Unknown method!")

    # Remaining shots distributed evenly amongst the terms that already have shots allocated to them
    while True:
        remaining_shots = n_shots - sum(shot_allocation)
        if not remaining_shots:
            break
        shot_allocation += np.array(
            [1 if (_i < remaining_shots and shots) else 0 for _i, shots in enumerate(shot_allocation)]
        )
    return shot_allocation.tolist()


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
            total = sum(pauli_term_sample_expectation(circuit, term, n_shots) for term in hamiltonian.terms)
        else:
            # Define shot_allocation list if not given
            if shot_allocation is None:
                shot_allocation = allocate_shots(hamiltonian, n_shots)
            # Then sum up the individual Pauli terms in the Hamiltonian to get the overall expectation value
            total = sum(
                pauli_term_sample_expectation(circuit, term, shots)
                for shots, term in zip(shot_allocation, hamiltonian.terms)
            )
        # Add the constant term if present. Note: Energies (in chemistry) are all real values
        total += hamiltonian.constant.real
        return total

    # Expectation value from state vector simulation
    result = circuit(nshots=1)
    state_ket = result.state()
    return hamiltonian.expectation(state_ket)
