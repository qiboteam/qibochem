"""
Functions for obtaining the expectation value for some given circuit and Hamiltonian, either from a state
vector simulation, or from sample measurements
"""

from functools import reduce

import qibo
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z
from sympy import Add, Mul
from sympy.core.numbers import One

from qibochem.measurement.optimization import measurement_basis_rotations
from qibochem.measurement.shot_allocation import (
    allocate_shots,
    allocate_shots_by_variance,
)


def expectation(circuit: qibo.models.Circuit, hamiltonian: qibo.hamiltonians.SymbolicHamiltonian):
    """
    Expectation value using state vector simulations

    Args:
        circuit (:class:`qibo.models.Circuit`): Quantum circuit ansatz
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Molecular Hamiltonian

    Returns:
        float: Expectation value of the Hamiltonian for the given circuit
    """
    result = circuit()
    state_ket = result.state()
    return hamiltonian.expectation(state_ket)


def constant_term(hamiltonian):
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


def pauli_term_measurement_expectation(expression, frequencies, qubit_map):
    """
    Calculate the expectation value of group of general Pauli strings for some measurement frequencies

    Args:
        expression (sympy.Expr): (Group of) Pauli terms, e.g. X0*Z2 + Y1
        frequencies: Measurement frequencies, taken from MeasurementOutcomes.frequencies(binary=True)
        qubit_map (dict): Mapping the output frequencies to the corresponding qubit

    Returns:
        float: Expectation value of expression
    """
    z_only_ham = None  # Needed to satisfy pylint :(
    if isinstance(expression, Add):
        # Sum of multiple Pauli terms
        return sum(pauli_term_measurement_expectation(term, frequencies, qubit_map) for term in expression.args)
    if isinstance(expression, Mul):
        # Single Pauli term
        pauli_z_terms = [Z(term.target_qubit) if isinstance(term, (X, Y, Z)) else term for term in expression.args]
        z_only_ham = SymbolicHamiltonian(reduce(lambda x, y: x * y, pauli_z_terms, 1.0))
    elif isinstance(expression, (X, Y, Z)):
        z_only_ham = SymbolicHamiltonian(Z(expression.target_qubit), nqubits=expression.target_qubit + 1)
    # Can now apply expectation_from_samples directly
    return z_only_ham.expectation_from_samples(frequencies, qubit_map=qubit_map)


def expectation_from_samples(
    circuit: qibo.models.Circuit,
    hamiltonian: SymbolicHamiltonian,
    n_shots: int = 1000,
    grouping=None,
    n_shots_per_pauli_term: bool = True,
    shot_allocation=None,
) -> float:
    """
    Calculate expectation value of some Hamiltonian using sample measurements from running a Qibo quantum circuit

    Args:
        circuit (:class:`qibo.models.Circuit`): Quantum circuit ansatz
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Molecular Hamiltonian
        n_shots (int): Number of times the circuit is run. Default: ``1000``
        grouping (str): Whether or not to group Hamiltonian terms together to reduce the measurement
            cost. Available options: ``None``: (Default) No grouping of Hamiltonian terms, and
            ``"qwc"``: Terms that commute qubitwise are grouped together
        n_shots_per_pauli_term (bool): Whether or not ``n_shots`` is used for each Pauli term (or group of terms) in the
            Hamiltonian, or for *all* the (group of) terms in the Hamiltonian. Default: ``True``; ``n_shots`` are used
            to get the expectation value for each term (group of terms) in the Hamiltonian.
        shot_allocation (list): Iterable containing the number of shots to be allocated to each term (or group of terms) in the
            Hamiltonian respectively if the `n_shots_per_pauli_term` argument is ``False``. Default: ``None``; shots are
            allocated based on the magnitudes of the coefficients of the Hamiltonian terms.

    Returns:
        float: Hamiltonian expectation value for the given circuit using sample measurements
    """
    # Group up Hamiltonian terms to reduce the measurement cost
    grouped_terms = measurement_basis_rotations(hamiltonian, grouping=grouping)

    # Check shot_allocation argument if not using n_shots_per_pauli_term
    if not n_shots_per_pauli_term:
        if shot_allocation is None:
            shot_allocation = allocate_shots(grouped_terms, n_shots)
        assert len(shot_allocation) == len(
            grouped_terms
        ), f"shot_allocation list ({len(shot_allocation)}) doesn't match the number of grouped terms ({len(grouped_terms)})"

    total = constant_term(hamiltonian)
    for _i, (expression, measurement_gates) in enumerate(grouped_terms):
        _circuit = circuit.copy()
        _circuit.add(measurement_gates)

        # Number of shots used to run the circuit depends on n_shots_per_pauli_term
        result = _circuit(nshots=n_shots if n_shots_per_pauli_term else shot_allocation[_i])

        frequencies = result.frequencies(binary=True)
        if frequencies:  # Needed because might have cases whereby no shots allocated to a group
            qubit_map = sorted(qubit for gate in measurement_gates for qubit in gate.target_qubits)
            total += pauli_term_measurement_expectation(expression, frequencies, qubit_map)
    return total


def sample_statistics(circuit, grouped_terms, n_shots=1000, grouping=None):
    """
    An alternative to `expectation_from_samples` to be used when both the expectation values and sample variances are of
    interest. Unlike `expectation_from_samples`, this function does not have the flexibility of allocating shots
    specifically to each term (group) in the Hamiltonian; a fixed number of shots (`n_shots`) will be allocated to each
    term (group) instead.

    Args:
        circuit (:class:`qibo.models.Circuit`): Quantum circuit ansatz
        grouped_terms (list): List of two-tuples; the first item in the tuple is a group of Pauli terms
            (:class:`sympy.Expr`), and the second is a list of measurement gates (:class:`qibo.gates.M`) that can be
            used to get the expectation value for the corresponding expression
        n_shots (int): Number of times the circuit is run for each Hamiltonian term (group). Default: ``1000``
        grouping (str): Whether or not to group Hamiltonian terms together to reduce the measurement
            cost. Available options: ``None``: (Default) No grouping of Hamiltonian terms, and
            ``"qwc"``: Terms that commute qubitwise are grouped together

    Returns:
        list: Sample expectation values for each Hamiltonian term (group) with respect to the given circuit
        list: Sample variances for each Hamiltonian term (group) with respect to the given circuit
    """
    expectation_values, expectation_variances = [], []
    for expression, measurement_gates in grouped_terms:
        _circuit = circuit.copy()
        _circuit.add(measurement_gates)
        result = _circuit(nshots=n_shots)
        frequencies = result.frequencies(binary=True)
        qubit_map = sorted(qubit for gate in measurement_gates for qubit in gate.target_qubits)
        # Calculate sample mean first, then iterate through the obtained result frequencies to get the sample variance
        sample_mean = pauli_term_measurement_expectation(expression, frequencies, qubit_map)
        sample_variance = sum(
            (pauli_term_measurement_expectation(expression, {freq: count}, qubit_map) - sample_mean) ** 2
            for freq, count in frequencies.items()
        ) / (n_shots - 1)
        expectation_values.append(sample_mean)
        expectation_variances.append(sample_variance)
    return expectation_values, expectation_variances


def expectation_variance(circuit, hamiltonian, n_trial_shots, grouping):
    """
    Calculate the variance in the expectation value of a single Hamiltonian term group
    """
    # TODO: This is ugly...!!! How to improve upon it?
    sample_results = [
        expectation_from_samples(circuit, hamiltonian, n_shots=1, grouping=grouping) for _ in range(n_trial_shots)
    ]
    sample_mean = sum(sample_results) / n_trial_shots
    sample_variance = sum((_x - sample_mean) ** 2 for _x in sample_results) / (n_trial_shots - 1)
    return sample_mean, sample_variance


def v_expectation(circuit, hamiltonian, n_shots, n_trial_shots, grouping=None, method="vmsa"):
    """
    Loss function for finding the expectation value of a Hamiltonian using shots. Shots are allocated according to the
    Variance-Minimized Shot Assignment (VMSA) or Variance-Preserved Shot Reduction (VPSR) approaches suggested in the
    reference paper (see below). Essentially, a uniform number of trial shots are first used to find the sample variance
    for each term in the Hamiltonian. For the VMSA method, all the remaining shots after allocated after obtaining the
    sample variance of each Hamiltonian term (group), while for the VPSR method, a sufficient number of shots are
    allocated to each term (group) to keep their variance below a certain threshold, i.e. not all shots are allocated.

    Args:
        circuit (:class:`qibo.models.Circuit`): Circuit ansatz for running VQE
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Hamiltonian of interest
        n_shots (int): Total number of shots for finding the Hamiltonian expectation value
        n_trial_shots (int): Number of shots to use for finding the sample variance for each Hamiltonian term
        grouping (str): Whether or not to group Hamiltonian terms together. Available options: ``None``: (Default) No
            grouping of Hamiltonian terms, and ``"qwc"``: Terms that commute qubitwise are grouped together
        method (str): Which variance-based method to use; must be either `"vmsa"` (default) or `"vpsr"`.

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
    grouped_terms = measurement_basis_rotations(hamiltonian, grouping=grouping)
    # Input check: n_trial_shots * nH terms <= n_shots
    assert (
        n_trial_shots * len(grouped_terms) <= n_shots
    ), f"n(Trial shots = {n_trial_shots}) * n(Term groups = {len(grouped_terms)}) > n(Total shots = {n_shots})"
    # Sample means and variances for each term group, using n_trial_shots
    # sample_results = [
    #     expectation_variance(circuit, SymbolicHamiltonian(expression), n_trial_shots, grouping=grouping)
    #     for expression, _ in grouped_terms
    # ]
    sample_means, sample_variances = sample_statistics(circuit, grouped_terms, n_shots=n_trial_shots, grouping=grouping)
    # sample_means = [result[0] for result in sample_results]
    # sample_variances = [result[1] for result in sample_results]
    # Assign remaining (n_shots - nH terms * n_trial_shots) based on the computed sample variances
    remaining_shot_allocation = allocate_shots_by_variance(n_shots, n_trial_shots, sample_variances, method=method)
    new_mean_values = [
        expectation_from_samples(circuit, SymbolicHamiltonian(expression), n_shots=_n, grouping=grouping)
        for (expression, _), _n in zip(grouped_terms, remaining_shot_allocation)
    ]
    # Combine the results from the initial n_trial_shots and the remaining shots
    sum_values = [
        n_trial_shots * initial_mean + _n * new_mean
        for initial_mean, _n, new_mean in zip(sample_means, remaining_shot_allocation, new_mean_values)
    ]
    final_mean_values = [value / (n_trial_shots + _n) for value, _n in zip(sum_values, remaining_shot_allocation)]
    return sum(final_mean_values) + constant_term(hamiltonian)
