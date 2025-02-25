"""
Functions for obtaining the expectation value for some given circuit and Hamiltonian, either from a state
vector simulation, or from sample measurements
"""

from functools import reduce

import qibo
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z
from sympy.core.numbers import One

from qibochem.measurement.optimization import (
    allocate_shots,
    measurement_basis_rotations,
)


def expectation(circuit: qibo.models.Circuit, hamiltonian: SymbolicHamiltonian):
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


def pauli_term_measurement_expectation(hamiltonian, frequencies, qubit_map):
    """
    Calculate the expectation value of a single general Pauli string for some measurement frequencies

    Args:
        hamiltonian (SymbolicHamiltonian): Single general Pauli term, e.g. X0*Z2
        frequencies: Measurement frequencies, taken from MeasurementOutcomes.frequencies(binary=True)

    Returns:
        float: Expectation value of pauli_term
    """
    z_terms = 0.0
    # Replace every (non-I) Symbol with Z, then include the term coefficient
    for term, coefficient in hamiltonian.form.as_coefficients_dict().items():
        # if isinstance(term, One): # Return any constant term immediately
        #     z_terms += coefficient
        # else:
        if True:
            if isinstance(term, (X, Y, Z)):
                z_terms += coefficient * Z(term.target_qubit)
            else:
                pauli_z_terms = [
                    Z(factor.target_qubit) if isinstance(factor, (X, Y, Z)) else factor for factor in term.args
                ]
                z_terms += reduce(lambda x, y: x * y, pauli_z_terms, 1.0)

    # if not hamiltonian.form.args:
    #     z_only_ham = SymbolicHamiltonian(Z(hamiltonian.form.target_qubit))
    # else:
    #     pauli_z_terms = [
    #         Z(term.target_qubit) if isinstance(term, (X, Y, Z)) else term
    #         for term in hamiltonian.form.args
    #     ]
    #     # pauli_z = [Z(int(factor.target_qubit)) for factor in pauli_term.factors if factor.name[0] != "I"]
    #     z_only_ham = SymbolicHamiltonian(reduce(lambda x, y: x * y, pauli_z_terms, 1.0))
    # Can now apply expectation_from_samples directly
    # return z_only_ham.expectation_from_samples(frequencies, qubit_map=qubit_map)
    return SymbolicHamiltonian(z_terms).expectation_from_samples(frequencies, qubit_map=qubit_map)


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
        circuit (:class:`qibo.models.Circuit`): Quantum circuit ansatz
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Molecular Hamiltonian
        n_shots (int): Number of times the circuit is run. Default: ``1000``
        group_pauli_terms (str): Whether or not to group Hamiltonian terms together to reduce the measurement
            cost. Available options: ``None``: (Default) No grouping of Hamiltonian terms, and
            ``"qwc"``: Terms that commute qubitwise are grouped together
        n_shots_per_pauli_term (bool): Whether or not ``n_shots`` is used for each Pauli term (or group of terms) in the
            Hamiltonian, or for *all* the (group of) terms in the Hamiltonian. Default: ``True``; ``n_shots`` are used
            to get the expectation value for each term (group of terms) in the Hamiltonian.
        shot_allocation: Iterable containing the number of shots to be allocated to each term (or group of terms) in the
            Hamiltonian respectively if the `n_shots_per_pauli_term` argument is ``False``. Default: ``None``; shots are
            allocated based on the magnitudes of the coefficients of the Hamiltonian terms.

    Returns:
        float: Hamiltonian expectation value for the given circuit using sample measurements
    """
    # Group up Hamiltonian terms to reduce the measurement cost
    grouped_terms = measurement_basis_rotations(hamiltonian, grouping=group_pauli_terms)
    print(grouped_terms)

    # Check shot_allocation argument if not using n_shots_per_pauli_term
    if not n_shots_per_pauli_term:
        if shot_allocation is None:
            shot_allocation = allocate_shots(grouped_terms, n_shots)
        assert len(shot_allocation) == len(
            grouped_terms
        ), f"shot_allocation list ({len(shot_allocation)}) doesn't match the number of grouped terms ({len(grouped_terms)})"

    total = 0.0
    for _i, (ham_group, measurement_gates) in enumerate(grouped_terms):
        if measurement_gates:
            _circuit = circuit.copy()
            _circuit.add(measurement_gates)

            print(_circuit.draw())

            # Number of shots used to run the circuit depends on n_shots_per_pauli_term
            result = _circuit(nshots=n_shots if n_shots_per_pauli_term else shot_allocation[_i])

            frequencies = result.frequencies(binary=True)
            qubit_map = sorted(qubit for gate in measurement_gates for qubit in gate.target_qubits)
            if frequencies:  # Needed because might have cases whereby no shots allocated to a group
                # print("hamiltonian.form:", hamiltonian.form)
                # print(hamiltonian.form.as_coefficients_dict())
                # print("ham.form.args:", hamiltonian.form.args)
                total += sum(
                    pauli_term_measurement_expectation(SymbolicHamiltonian(coefficient * term), frequencies, qubit_map)
                    for term, coefficient in hamiltonian.form.as_coefficients_dict().items()
                )

                # if hamiltonian.form.args:
                #     for term in hamiltonian.form.args:
                #         print(term)
                #         print(pauli_term_measurement_expectation(SymbolicHamiltonian(term), frequencies, qubit_map))

                #     total += sum(
                #         pauli_term_measurement_expectation(SymbolicHamiltonian(term), frequencies, qubit_map)
                #         for term in ham_group.form.args if isinstance(term, Expr)
                #     )
                # Single Pauli

    # Add the constant term if present. Note: Energies (in chemistry) are all real values
    # total += hamiltonian.constant.real
    return total
