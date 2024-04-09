import numpy as np
from qibo import gates


def check_terms_commutativity(term1, term2, qubitwise=False):
    """
    Check if terms 1 and 2 are mutually commuting. The 'qubitwise' flag determines if the check is for general
    commutativity (False), or the stricter qubitwise commutativity.

    Args:
        term1/term2: Lists of strings representing a single Pauli term. E.g. ["X0", "Z1", "Y3"]. Obtained from a Qibo
            SymbolicTerm as ``[factor.name for factor in term.factors]``.
        qubitwise: Determines if the check is for general commutativity, or the stricter qubitwise commutativity

    Returns:
        bool: Do terms 1 and 2 commute?
    """
    # Get a list of common qubits for each term
    common_qubits = sorted(
        {_term[1:] for _term in term1 if _term[0] != "I"} & {_term[1:] for _term in term2 if _term[0] != "I"}
    )
    if not common_qubits:
        return True
    # Get the single Pauli operators for the common qubits for both Pauli terms
    term1_ops = [_op for _op in term1 if _op[1:] in common_qubits]
    term2_ops = [_op for _op in term2 if _op[1:] in common_qubits]
    if qubitwise:
        # Qubitwise: Compare the Pauli terms at the common qubits. Any difference => False
        return all(_op1 == _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # General commutativity:
    # Get the number of single Pauli operators that do NOT commute
    n_noncommuting_ops = sum(_op1 != _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # term1 and term2 have general commutativity iff n_noncommuting_ops is even
    return n_noncommuting_ops % 2 == 0


def measurement_basis_rotations(hamiltonian, n_qubits, grouping=None):
    """
    Split up and sort the Hamiltonian terms to get the basis rotation gates to be applied to a quantum circuit for the
    respective (group of) terms in the Hamiltonian

    Args:
        hamiltonian (SymbolicHamiltonian): Hamiltonian (that only contains X/Y terms?)
        n_qubits: Number of qubits in the quantum circuit.
        grouping: Whether or not to group the X/Y terms together, i.e. use the same set of measurements to get the expectation
            values of a group of terms simultaneously. Default value of ``None`` will not group any terms together, which is
            the only option currently implemented.

    Returns:
        list: List of two-tuples, with each tuple given as ([`list of measurement gates`], [term1, term2, ...]), where
        term1, term2, ... are SymbolicTerms. The first tuple always corresponds to all the Z terms present, which will be two
        empty lists - ``([], [])`` - if there are no Z terms present.
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
        result.append(([], []))  # No terms with only Z's
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
    Allocate shots to each group of terms in the Hamiltonian for calculating the expectation value of the Hamiltonian.

    Args:
        grouped_terms (list): Output of measurement_basis_rotations(hamiltonian, n_qubits, grouping=None
        n_shots (int): Total number of shots to be allocated
        method (str): How to allocate the shots. The available options are: ``"c"``/``"coefficients"``: ``n_shots`` is distributed
            based on the relative magnitudes of the term coefficients, ``"u"``/``"uniform"``: ``n_shots`` is distributed evenly
            amongst each term. Default value: ``"c"``.
        max_shots_per_term (int): Upper limit for the number of shots allocated to an individual group of terms. If not given,
            will be defined as a fraction (largest coefficient over the sum of all coefficients in the Hamiltonian) of ``n_shots``.

    Returns:
        list: A list containing the number of shots to be used for each group of Pauli terms respectively.
    """
    if method is None:
        method = "c"
    if max_shots_per_term is None:
        # Define based on the fraction of the term group with the largest coefficients w.r.t. sum of all coefficients.
        term_coefficients = np.array(
            [sum(abs(term.coefficient.real) for term in terms) for (_, terms) in grouped_terms]
        )
        max_shots_per_term = int(np.ceil(n_shots * (np.max(term_coefficients) / sum(term_coefficients))))
        # max_shots_per_term = min(max_shots_per_term, 250)  #  Is there an optimal value - Explore further?
    max_shots_per_term = min(n_shots, max_shots_per_term)  # Don't let max_shots_per_term > n_shots if manually defined

    n_terms = len(grouped_terms)
    shot_allocation = np.zeros(n_terms, dtype=int)

    while True:
        remaining_shots = n_shots - sum(shot_allocation)
        if not remaining_shots:
            break

        # In case n_shots is too high s.t. max_shots_per_term is too low
        # Increase max_shots_per_term, and redo the shot allocation
        if np.min(shot_allocation) == max_shots_per_term:
            max_shots_per_term = min(2 * max_shots_per_term, n_shots)
            shot_allocation = np.zeros(n_terms, dtype=int)
            continue

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
            if _shot_allocation.any():
                term_coefficients /= sum(term_coefficients)
                _shot_allocation = (remaining_shots * term_coefficients).astype(int)
            else:
                # For distributing the remaining few shots, i.e. remaining_shots << n_terms
                _shot_allocation = np.array(
                    allocate_shots(grouped_terms, remaining_shots, max_shots_per_term=remaining_shots, method="u")
                )

        elif method in ("u", "uniform"):
            # Uniform distribution of shots for every term. Extra shots are randomly distributed
            _shot_allocation = np.array([remaining_shots // n_terms for _ in range(n_terms)])
            if not _shot_allocation.any():
                _shot_allocation = np.zeros(n_terms)
                _shot_allocation[:remaining_shots] = 1
                np.random.shuffle(_shot_allocation)

        else:
            raise NameError("Unknown method!")

        # Add on to the current allocation, and set upper limit to the number of shots for a given term
        shot_allocation += _shot_allocation.astype(int)
        shot_allocation = np.clip(shot_allocation, 0, max_shots_per_term)

    return shot_allocation.tolist()
