"""
Functions for allocating shots to the Hamiltonian terms
"""

import numpy as np
from sympy.core.numbers import One


def coefficients_sum(expression):
    """Sum up the absolute value of the coefficients for all non-constant terms in a sympy.Expr"""
    return sum(abs(coeff) for term, coeff in expression.as_coefficients_dict().items() if not isinstance(term, One))


def allocate_shots(grouped_terms, n_shots, method=None, max_shots_per_term=None):
    """
    Allocate shots to each group of terms in the Hamiltonian for calculating the expectation value of the Hamiltonian.

    Args:
        grouped_terms (list): Output of measurement_basis_rotations; list of two-tuples with the first term a
            :class:`sympy.Expr` and the second the list of corresponding measurement gates (not used here).
        n_shots (int): Total number of shots to be allocated
        method (str): How to allocate the shots. The available options are: ``"c"``/``"coefficients"``: ``n_shots`` is
            distributed based on the relative magnitudes of the term coefficients, ``"u"``/``"uniform"``: ``n_shots``
            is distributed evenly amongst each term. Default value: ``"c"``.
        max_shots_per_term (int): Upper limit for the number of shots allocated to an individual group of terms. If not
            given, will be defined as a fraction (largest coefficient over the sum of all coefficients in the
            Hamiltonian) of ``n_shots``.

    Returns:
        list: A list containing the number of shots to be used for each group of Pauli terms respectively.
    """
    if method is None:
        method = "c"
    if max_shots_per_term is None:
        # Define based on the fraction of the term group with the largest coefficients w.r.t. sum of all coefficients.
        # Using coefficients**(2/3) following arxiv:2307.06504
        term_coefficients = np.array(
            [coefficients_sum(expression) ** (2 / 3) for (expression, _) in grouped_terms],
            dtype=float,
        )
        max_shots_per_term = int(np.ceil(n_shots * (np.max(term_coefficients) / sum(term_coefficients))))
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
            # Using coefficients**(2/3) following arxiv:2307.06504
            term_coefficients = np.array(
                [
                    coefficients_sum(expression) ** (2 / 3) if shots < max_shots_per_term else 0.0
                    for shots, (expression, _) in zip(shot_allocation, grouped_terms)
                ],
                dtype=float,
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


def allocate_shots_by_variance(total_shots, n_trial_shots, variance_values, method="vmsa"):
    """
    Allocate shots for each term in a Hamiltonian based on the computed sample variances of each term.

    Args:
        total_shots (int): Total shot budget for each expectation value evaluation
        n_trial_shots (int): Number of shots used to obtain the sample variances of each term group
        variance_values (List[float]): Sample variances for each term group

    Returns:
        list: List of integers corresponding to the allocation of the remaining shots
    """
    assert method in ("vmsa", "vpsr"), f"Unknown shot assignment method ({method}) called"
    n_groups = len(variance_values)
    remaining_shots = total_shots - n_groups * n_trial_shots
    std_dev_values = [_var**0.5 for _var in variance_values]
    # eta in Equation 17 of the reference paper, equal to 1 for VMSA, <1 if VPSR
    _eta = 1 if method == "vmsa" else sum(std_dev_values) ** 2 / (n_groups * sum(variance_values))
    # Calculate everything as floats first, then convert to ints
    allocated_shots = [int(_eta * std_dev * remaining_shots / sum(std_dev_values)) for std_dev in std_dev_values]
    # Throw any leftover shots into the last term (arbitrarily) if using VMSA
    if method == "vmsa":
        allocated_shots[-1] += remaining_shots - sum(allocated_shots)
    return allocated_shots
