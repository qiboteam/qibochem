from functools import reduce

import numpy as np
import qibo
from qibo import gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z


def allocate_shots(grouped_terms, n_shots=1000, method=None, max_shots_per_term=500, threshold=0.05):
    """
    Allocate shots to each group of terms in the Hamiltonian for the calculation of the expectation value
    TODO: Clean up documentation!

    Args:
        grouped_terms (list): Output of measurement_basis_rotations(hamiltonian, n_qubits, grouping=None
        n_shots (int): Total number of shots to be allocated. Default: ``1000``
        method (str): How to allocate the shots. The available options are: ``c``/``coefficients``: ``n_shots`` is distributed
            based on the relative magnitudes of the term coefficients, ``u``/``uniform``: ``n_shots`` is distributed evenly
            amongst each term.
        max_shots_per_term (int): Upper limit for the number of shots allocated to an individual term
        threshold: Used in the ``coefficients`` method to decide which terms to ignore, i.e. no shots will be allocated to
            terms with coefficient < threshold*np.max(term_coefficients)

    Returns:
        shot_allocation: A list containing the number of shots to be used for each group of Pauli terms respectively
    """
    if method is None:
        method = "c"
    n_terms = len(grouped_terms)
    shot_allocation = np.zeros(n_terms, dtype=int)

    while True:
        remaining_shots = n_shots - sum(shot_allocation)
        if not remaining_shots:
            break

        if method in ("c", "coefficients"):
            pass

        elif method in ("u", "uniform"):
            # Even distribution of shots for every group of Pauli terms
            _shot_allocation = np.array([remaining_shots // n_terms for _ in range(n_terms)])
            if not _shot_allocation.any():  # In case n_shots < n_terms
                _shot_allocation = np.array(
                    [1 if _i < remaining_shots else 0 for _i, _shots in enumerate(shot_allocation)]
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

    # if method in ("c", "coefficients"):
    #     # Split shots based on the relative magnitudes of the coefficients of the Pauli terms
    #     term_coefficients = np.array([abs(term.coefficient.real) for term in hamiltonian.terms])
    #     # Only keep terms that are at least threshold*largest_coefficient
    #     # Element-wise multiplication of the mask
    #     term_coefficients *= term_coefficients > threshold * np.max(term_coefficients)
    #     term_coefficients /= sum(term_coefficients)  # Normalise term_coefficients
    #     shot_allocation = (n_shots * term_coefficients).astype(int)
    # elif method in ("u", "uniform"):
    #     # Split evenly amongst all the terms in the Hamiltonian
    #     shot_allocation = np.array([n_shots // n_terms for _ in range(n_terms)])
    # else:
    #     raise NameError("Unknown method!")

    # # Remaining shots distributed evenly amongst the terms that already have shots allocated to them
    #     remaining_shots = n_shots - sum(shot_allocation)
    #     if not remaining_shots:
    #         break
    #     shot_allocation += np.array(
    #         [1 if (_i < remaining_shots and shots) else 0 for _i, shots in enumerate(shot_allocation)]
    #     )

    return shot_allocation.astype(int).tolist()


def symbolic_term_to_symbol(symbolic_term):
    """Convert a single Pauli word in the form of a Qibo SymbolicTerm to a Qibo Symbol"""
    return symbolic_term.coefficient * reduce(lambda x, y: x * y, symbolic_term.factors, 1.0)


def pauli_term_measurement_expectation(pauli_term, frequencies):
    """
    Calculate the expectation value of a single general Pauli string for some measurement frequencies

    Args:
        pauli_term (SymbolicTerm): Single general Pauli term, e.g. X0*Z2
        frequencies: Measurement frequencies, taken from MeasurementOutcomes.frequencies(binary=True)

    Returns:
        Expectation value of the pauli_term (float)
    """
    # Replace every (non-I) Symbol with Z, then include the term coefficient
    pauli_z = [Z(int(factor.target_qubit)) for factor in pauli_term.factors if factor.name[0] != "I"]
    z_only_ham = SymbolicHamiltonian(pauli_term.coefficient * reduce(lambda x, y: x * y, pauli_z, 1.0))
    # Can now apply expectation_from_samples directly
    return z_only_ham.expectation_from_samples(frequencies)


def measurement_basis_rotations(hamiltonian, n_qubits, grouping=None):
    """
    Split up and sort the Hamiltonian terms to get the basis rotation gates to be applied to a quantum circuit for the
    respective (group of) terms in the Hamiltonian

    Args:
        hamiltonian (SymbolicHamiltonian): Hamiltonian (that only contains X/Y terms?)
        n_qubits: Number of qubits in the quantum circuit.
        grouping: Whether or not to group the X/Y terms together, i.e. use the same set of measurements to get the expectation
            values of a group of terms simultaneously. Default value of ``None`` will not group any terms together

    Returns:
        List of two-tuples, with each tuple given as ([`list of measurement gates`], [term1, term2, ...]),
        where term1, term2, ... are SymbolicTerms. The first term always corresponds to all the Z terms present.
    """
    result = []
    # Split up the Z and X/Y terms first
    z_only_terms = [
        term for term in hamiltonian.terms if not any(factor.name[0] in ("X", "Y") for factor in term.factors)
    ]
    xy_terms = [term for term in hamiltonian.terms if term not in z_only_terms]
    # Add the Z terms into result first
    if z_only_terms:
        result.append(([gates.M(_i) for _i in range(n_qubits)], z_only_terms))
    else:
        result.append(([], []))
    # Then add the X/Y terms in
    if xy_terms:
        if grouping is None:
            result += [
                (
                    [
                        gates.M(int(factor.target_qubit), basis=type(factor.gate))
                        for factor in term.factors
                        if factor.name[0] != "I"
                    ],
                    [
                        term,
                    ],
                )
                for term in xy_terms
            ]
        else:
            raise NotImplementedError("Not ready yet!")
    return result


def expectation(
    circuit: qibo.models.Circuit,
    hamiltonian: SymbolicHamiltonian,
    from_samples: bool = False,
    n_shots: int = 1000,
    group_pauli_terms=None,
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
        group_pauli_terms: Whether or not to group Pauli X/Y terms in the Hamiltonian together to reduce the measurement cost
            TODO: Draft code and documentation!
        n_shots_per_pauli_term (Boolean): Whether or not ``n_shots`` is used for each Pauli term in the Hamiltonian, or for
            *all* the terms in the Hamiltonian. Default: ``True``; ``n_shots`` are used to get the expectation value for each
            term in the Hamiltonian.
        shot_allocation: Iterable containing the number of shots to be allocated to each term in the Hamiltonian respectively.
            Default: ``None``; then the ``allocate_shots`` function is called to build the list.

    Returns:
        Hamiltonian expectation value (float)
    """
    if from_samples:
        # (Eventually) measurement_basis_rotations will be used to group up some terms so that one
        # set of measurements can be used for multiple X/Y terms
        grouped_terms = measurement_basis_rotations(hamiltonian, circuit.nqubits, grouping=group_pauli_terms)

        # Check shot_allocation argument if not using n_shots_per_pauli_term
        if not n_shots_per_pauli_term:
            if shot_allocation is None:
                shot_allocation = allocate_shots(grouped_terms, n_shots)
            assert len(shot_allocation) == len(
                grouped_terms
            ), "shot_allocation list must be the same size as the number of grouped terms!"

        total = 0.0
        for _i, (measurement_gates, terms) in enumerate(grouped_terms):
            if measurement_gates and terms:
                _circuit = circuit.copy()
                _circuit.add(measurement_gates)

                # Number of shots used to run the circuit depends on n_shots_per_pauli_term
                result = _circuit(nshots=n_shots if n_shots_per_pauli_term else shot_allocation[_i])

                frequencies = result.frequencies(binary=True)
                if frequencies:  # Needed because might have cases whereby no shots allocated to a group
                    # First term is all Z terms, can use expectation_from_samples directly.
                    # Otherwise, need to use the general pauli_term_measurement_expectation function
                    if _i > 0:
                        total += sum(pauli_term_measurement_expectation(term, frequencies) for term in terms)
                    else:
                        z_ham = SymbolicHamiltonian(sum(symbolic_term_to_symbol(term) for term in terms))
                        total += z_ham.expectation_from_samples(frequencies)
        # Add the constant term if present. Note: Energies (in chemistry) are all real values
        total += hamiltonian.constant.real
        return total

    # Expectation value from state vector simulation
    result = circuit(nshots=1)
    state_ket = result.state()
    return hamiltonian.expectation(state_ket)
