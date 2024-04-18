from functools import reduce

import qibo
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z

from qibochem.measurement.optimization import (
    allocate_shots,
    measurement_basis_rotations,
)


def expectation(circuit: qibo.models.Circuit, hamiltonian: SymbolicHamiltonian):
    """
    Expectation value using state vector simulations
    TODO: Docstring

    """
    # Expectation value from state vector simulation
    result = circuit(nshots=1)
    state_ket = result.state()
    return hamiltonian.expectation(state_ket)


def symbolic_term_to_symbol(symbolic_term):
    """Convert a single Pauli word in the form of a Qibo SymbolicTerm to a Qibo Symbol"""
    return symbolic_term.coefficient * reduce(lambda x, y: x * y, symbolic_term.factors, 1.0)


def pauli_term_measurement_expectation(pauli_term, frequencies, qubit_map):
    """
    Calculate the expectation value of a single general Pauli string for some measurement frequencies

    Args:
        pauli_term (SymbolicTerm): Single general Pauli term, e.g. X0*Z2
        frequencies: Measurement frequencies, taken from MeasurementOutcomes.frequencies(binary=True)

    Returns:
        float: Expectation value of pauli_term
    """
    # Replace every (non-I) Symbol with Z, then include the term coefficient
    pauli_z = [Z(int(factor.target_qubit)) for factor in pauli_term.factors if factor.name[0] != "I"]
    z_only_ham = SymbolicHamiltonian(pauli_term.coefficient * reduce(lambda x, y: x * y, pauli_z, 1.0))
    # Can now apply expectation_from_samples directly
    return z_only_ham.expectation_from_samples(frequencies, qubit_map=qubit_map)


def expectation_from_samples(
    circuit: qibo.models.Circuit,
    hamiltonian: SymbolicHamiltonian,
    n_shots: int = 1000,
    group_pauli_terms=None,
    n_shots_per_pauli_term: bool = True,
    shot_allocation=None,
) -> float:
    """
    Calculate expectation value of some Hamiltonian using sample measurements from running a Qibo quantum circuit

    Args:
        circuit (qibo.models.Circuit): Quantum circuit ansatz
        hamiltonian (qibo.hamiltonians.SymbolicHamiltonian): Molecular Hamiltonian
        n_shots (int): Number of times the circuit is run if ``from_samples=True``. Default: ``1000``
        group_pauli_terms: Whether or not to group Pauli X/Y terms in the Hamiltonian together to reduce the measurement cost.
            Default: ``None``; each of the Hamiltonian terms containing X/Y are in their own individual groups.
        n_shots_per_pauli_term (bool): Whether or not ``n_shots`` is used for each Pauli term in the Hamiltonian, or for
            *all* the terms in the Hamiltonian. Default: ``True``; ``n_shots`` are used to get the expectation value for each
            term in the Hamiltonian.
        shot_allocation: Iterable containing the number of shots to be allocated to each term in the Hamiltonian respectively if
            n_shots_per_pauli_term is ``False``. Default: ``None``; shots are allocated based on the magnitudes of the coefficients
            of the Hamiltonian terms.

    Returns:
        float: Hamiltonian expectation value
    """
    # From sample measurements:
    # (Eventually) measurement_basis_rotations will be used to group up some terms so that one
    # set of measurements can be used for multiple X/Y terms
    grouped_terms = measurement_basis_rotations(hamiltonian, grouping=group_pauli_terms)

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
            qubit_map = sorted(qubit for gate in measurement_gates for qubit in gate.target_qubits)
            if frequencies:  # Needed because might have cases whereby no shots allocated to a group
                # First term is all Z terms, can use expectation_from_samples directly.
                # Otherwise, need to use the general pauli_term_measurement_expectation function
                if _i > 0:
                    total += sum(pauli_term_measurement_expectation(term, frequencies, qubit_map) for term in terms)
                else:
                    z_ham = SymbolicHamiltonian(sum(symbolic_term_to_symbol(term) for term in terms))
                    qubit_map = sorted({factor.target_qubit for term in terms for factor in term.factors})
                    total += z_ham.expectation_from_samples(frequencies, qubit_map=qubit_map)
    # Add the constant term if present. Note: Energies (in chemistry) are all real values
    total += hamiltonian.constant.real
    return total
