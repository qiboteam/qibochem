"""
Functions for obtaining the expectation value for some given circuit and Hamiltonian, either from a state
vector simulation, or from sample measurements
"""

from collections import Counter
from functools import reduce

from qibo import Circuit
from qibo.gates import Gate
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z
from sympy import Add, Mul
from sympy.core.expr import Expr
from sympy.core.numbers import One

from qibochem.measurement.optimization import _measurement_basis_rotations
from qibochem.measurement.shot_allocation import (
    allocate_shots,
    allocate_shots_by_variance,
)


def _constant_term(hamiltonian: SymbolicHamiltonian) -> complex:
    """Extract the constant term (if any) from a given SymbolicHamiltonian"""
    constant = 0.0
    ham_form = hamiltonian.form
    if ham_form.args:
        # Hamiltonian has >1 term
        find_constant = [coeff for term, coeff in ham_form.as_coefficients_dict().items() if isinstance(term, One)]
        constant = find_constant[0] if find_constant else 0.0
    else:
        # Single term is either a Pauli operator or a float
        constant = float(ham_form) if not isinstance(ham_form, (X, Y, Z)) else 0.0
    return constant


def _pauli_term_measurement_expectation(expression: Expr, frequencies: Counter[str], qubit_map: list[int]) -> float:
    """Calculate expectation of an expression with >=1 non-diagonal terms for a given set of measurement frequencies"""
    z_only_ham = None  # Needed to satisfy pylint :(
    if isinstance(expression, Add):
        # Sum of multiple Pauli terms
        return sum(_pauli_term_measurement_expectation(term, frequencies, qubit_map) for term in expression.args)
    if isinstance(expression, Mul):
        # Single Pauli term
        pauli_z_terms = [Z(term.target_qubit) if isinstance(term, (X, Y, Z)) else term for term in expression.args]
        z_only_ham = SymbolicHamiltonian(
            reduce(lambda x, y: x * y, pauli_z_terms, 1.0),
            nqubits=max(term.target_qubit for term in expression.args if isinstance(term, (X, Y, Z))) + 1,
        )
    elif isinstance(expression, (X, Y, Z)):
        z_only_ham = SymbolicHamiltonian(Z(expression.target_qubit), nqubits=expression.target_qubit + 1)
    # Can now apply expectation_from_samples directly
    return z_only_ham.expectation_from_samples(frequencies, qubit_map=qubit_map)


def expectation_from_samples(
    circuit: Circuit,
    hamiltonian: SymbolicHamiltonian,
    n_shots: int = 1000,
    grouping: str | None = None,
    n_shots_per_pauli_term: bool = True,
    shot_allocation: list[int] | None = None,
) -> float:
    """
    Calculate expectation value of some Hamiltonian using sample measurements from running a Qibo quantum circuit

    Args:
        circuit (:class:`qibo.models.Circuit`):
            Quantum circuit ansatz
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`):
            Molecular Hamiltonian
        n_shots (int):
            Number of times the circuit is run. Default: ``1000``
        grouping (str | None):
            Whether to group Hamiltonian terms together to reduce the measurement cost

            Options:
                - ``None``: No grouping of Hamiltonian terms (Default)
                - ``"qwc"``: Qubit-wise commuting terms are grouped together
                - ``"gc"``: Generally commuting terms are measured simultaneously by adding additional gates following the formulation by `Gokhale et al. <https://ieeexplore.ieee.org/abstract/document/9248636/>`_,
                - ``"gc2"``: Same as ``"gc"``, but uses the circuit formulation by `Yen et al. <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00008>`_ instead
        n_shots_per_pauli_term (bool):
            If ``True`` (Default), uses ``n_shots`` per Pauli term (or group of terms) to calculate the expectation
            value
        shot_allocation (list[int]):
            Shot allocation per Pauli term (or group of terms) when ``n_shots_per_pauli_term`` is ``False``.

    Returns:
        float: Hamiltonian expectation value for the given circuit using sample measurements
    """
    # Group up Hamiltonian terms to reduce the measurement cost
    grouped_terms = _measurement_basis_rotations(hamiltonian, grouping=grouping)

    # Check shot_allocation argument if not using n_shots_per_pauli_term
    if not n_shots_per_pauli_term:
        if shot_allocation is None:
            shot_allocation = allocate_shots(grouped_terms, n_shots)
        if len(shot_allocation) != len(grouped_terms):
            raise ValueError(
                f"{len(shot_allocation) = } doesn't match the number of grouped terms ({len(grouped_terms)})!"
            )

    total = _constant_term(hamiltonian)
    for _i, (expression, measurement_gates, rotation_gates) in enumerate(grouped_terms):
        _circuit = circuit.copy()
        _circuit.add(rotation_gates)
        _circuit.add(measurement_gates)

        # Number of shots used to run the circuit depends on n_shots_per_pauli_term
        nshots = n_shots if n_shots_per_pauli_term else shot_allocation[_i]
        if nshots:
            result = _circuit(nshots=nshots)
            frequencies = result.frequencies(binary=True)
            if frequencies:  # Needed because might have cases whereby no shots allocated to a group
                qubit_map = [qubit for gate in measurement_gates for qubit in gate.target_qubits]
                total += _pauli_term_measurement_expectation(expression, frequencies, qubit_map)
    return total


def sample_statistics(
    circuit: Circuit, grouped_terms: list[tuple[Expr, list[Gate]]], n_shots: int = 1000
) -> tuple[list[float], list[float]]:
    """
    An alternative to the :ref:`expectation_from_samples<expectation-samples>` function when both the expectation values
    and sample variances are of interest. Unlike :ref:`expectation_from_samples<expectation-samples>`, this function
    does not have the flexibility of allocating shots specifically to each term (group) in the Hamiltonian; a fixed
    number of shots will be allocated to each term (group) instead.

    Args:
        circuit (:class:`qibo.models.Circuit`):
            Quantum circuit ansatz
        grouped_terms (list[tuple[Expr, list[Gate]]]):
            Groups of Pauli terms and their corresponding rotation gates.
        n_shots (int):
            Number of times the circuit is run for each Hamiltonian term (group). Default: ``1000``

    Returns:
        tuple[list[float], list[float]]:
            Sample means (expectation values) and variances for each Hamiltonian term (group)
    """
    expectation_values, expectation_variances = [], []
    for expression, measurement_gates, rotation_gates in grouped_terms:
        _circuit = circuit.copy()
        _circuit.add(rotation_gates)
        _circuit.add(measurement_gates)
        result = _circuit(nshots=n_shots)
        frequencies = result.frequencies(binary=True)
        qubit_map = sorted(qubit for gate in measurement_gates for qubit in gate.target_qubits)
        # Calculate sample mean first, then iterate through the obtained result frequencies to get the sample variance
        sample_mean = _pauli_term_measurement_expectation(expression, frequencies, qubit_map)
        sample_variance = sum(
            (_pauli_term_measurement_expectation(expression, {freq: count}, qubit_map) - sample_mean) ** 2
            for freq, count in frequencies.items()
        ) / (n_shots - 1)
        expectation_values.append(sample_mean)
        expectation_variances.append(sample_variance)
    return expectation_values, expectation_variances


def v_expectation(
    circuit: Circuit,
    hamiltonian: SymbolicHamiltonian,
    n_shots: int,
    n_trial_shots: int,
    grouping: str | None = None,
    method: str = "vmsa",
) -> float:
    """
    An alternative loss function for finding the expectation value of a Hamiltonian using shots. Shots are allocated
    according to the Variance-Minimized Shot Assignment (VMSA) or Variance-Preserved Shot Reduction (VPSR) approaches
    suggested in the reference paper (given below).

    Essentially, a uniform number of trial shots are first used to find the sample variance for each term (group) in the
    Hamiltonian. For the VMSA method, the remaining shots are all allocated to minimise the total variance (calculated
    as the sum of the variances), while for the VPSR method, a sufficient number of shots are allocated to each term
    (group) to keep their variance - and by extension, the total variance - below a certain threshold. Unlike in the
    VMSA method, the VPSR method does not allocate all of the remaining shots.

    Args:
        circuit (:class:`qibo.models.Circuit`): Circuit ansatz for running VQE
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Hamiltonian of interest
        n_shots (int): Total number of shots for finding the Hamiltonian expectation value
        n_trial_shots (int): Number of shots to use for finding the sample variance for each Hamiltonian term
        grouping (str | None): Whether to group Hamiltonian terms together. The available options: ``None``: (Default),
            ``"qwc"``, ``"gc"``, and ``"gc2"`` (see :func:`qibochem.measurement.util.expectation_from_samples` for
            details)
        method (str): Variance-based method to use; must be either `"vmsa"` (default) or `"vpsr"`.

    Returns:
        float: Hamiltonian expectation value obtained using a variance-based shot allocation scheme

    Reference:
        1. L. Zhu, S. Liang, C. Yang, X. Li, *Optimizing Shot Assignment in Variational Quantum Eigensolver
        Measurement*, Journal of Chemical Theory and Computation, 2024, 20, 2390-2403
        (`link <https://pubs.acs.org/doi/10.1021/acs.jctc.3c01113>`__)
    """
    # Input check: method is valid
    assert method in ("vmsa", "vpsr"), f"Unknown shot assignment method ({method}) called"
    # Split up Hamiltonian into individual (groups of) terms to get the variance of each term (group)
    grouped_terms = _measurement_basis_rotations(hamiltonian, grouping=grouping)
    # Input check: n_trial_shots * nH terms <= n_shots
    assert (
        n_trial_shots * len(grouped_terms) <= n_shots
    ), f"n(Trial shots = {n_trial_shots}) * n(Term groups = {len(grouped_terms)}) > n(Total shots = {n_shots})"
    # Sample means and variances for each term group, using n_trial_shots
    sample_means, sample_variances = sample_statistics(circuit, grouped_terms, n_shots=n_trial_shots)
    # Assign remaining (n_shots - nH terms * n_trial_shots) based on the computed sample variances
    remaining_shot_allocation = allocate_shots_by_variance(n_shots, n_trial_shots, sample_variances, method=method)
    new_mean_values = [
        expectation_from_samples(circuit, SymbolicHamiltonian(expression), n_shots=_n, grouping=grouping)
        for (expression, _, _), _n in zip(grouped_terms, remaining_shot_allocation)
    ]
    # Combine the results from the initial n_trial_shots and the remaining shots
    sum_values = [
        n_trial_shots * initial_mean + _n * new_mean
        for initial_mean, _n, new_mean in zip(sample_means, remaining_shot_allocation, new_mean_values)
    ]
    final_mean_values = [value / (n_trial_shots + _n) for value, _n in zip(sum_values, remaining_shot_allocation)]
    return sum(final_mean_values) + _constant_term(hamiltonian)
