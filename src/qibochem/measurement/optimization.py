import numpy as np
from qibo import gates


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
