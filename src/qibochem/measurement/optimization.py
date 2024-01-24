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


def allocate_shots(grouped_terms, n_shots, method=None, max_shots_per_term=None):
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

    Returns:
        shot_allocation: A list containing the number of shots to be used for each group of Pauli terms respectively
    """
    if method is None:
        method = "c"
    if max_shots_per_term is None:
        # Define based on the fraction of the term group with the largest coefficients w.r.t. sum of all coefficients.
        term_coefficients = np.array(
            [sum(abs(term.coefficient.real) for term in terms) for (_, terms) in grouped_terms]
        )
        max_shots_per_term = int(n_shots * (np.max(term_coefficients) / sum(term_coefficients)))
        max_shots_per_term = min(max_shots_per_term, 250)  #  Can be explored further?

    n_terms = len(grouped_terms)
    shot_allocation = np.zeros(n_terms, dtype=int)

    while True:
        remaining_shots = n_shots - sum(shot_allocation)
        if not remaining_shots:
            break

        if method in ("c", "coefficients"):
            # Split shots based on the relative magnitudes of the coefficients of the (group of) Pauli term(s)
            # and only for terms that haven't reached the upper limit yet
            term_coefficients = np.array(
                [
                    sum(abs(term.coefficient.real) for term in terms) if shots < max_shots_per_term else 0.0
                    for shots, (_, terms) in zip(shot_allocation, grouped_terms)
                ]
            )

            # Normalise term_coefficients, then get an initial distribution of remaining_shots
            term_coefficients /= sum(term_coefficients)
            _shot_allocation = (remaining_shots * term_coefficients).astype(int)
            # Only keep the terms with >0 shots allocated, renormalise term_coefficients, and distribute again
            term_coefficients *= _shot_allocation > 0
            _shot_allocation = (remaining_shots * term_coefficients).astype(int)

            # Check that not too many shots have been allocated
            if sum(_shot_allocation) > remaining_shots:
                indices_to_deduct = np.nonzero(shot_allocation)[: (sum(_shot_allocation) - remaining_shots)]
                np.add.at(_shot_allocation, indices_to_deduct, -1)

            # Check to see if shots could be allocated? For distributing the remaining few shots
            if not _shot_allocation.any():
                # No shots allocated, need to re-distribute the remaining shots
                # Check to see which terms have shots allocated, and haven't reached the max shot limit
                nonzero_indices = np.where((shot_allocation * (shot_allocation < max_shots_per_term)) != 0)[0]
                # Check if there are enough shots to allocate to the remaining terms equally
                if remaining_shots < len(nonzero_indices):
                    nonzero_indices = nonzero_indices[:remaining_shots]
                np.add.at(_shot_allocation, nonzero_indices, 1)

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

    return shot_allocation.astype(int).tolist()
