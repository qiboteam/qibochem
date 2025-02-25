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

from qibochem.measurement.optimization import (
    allocate_shots,
    constant_term,
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
    # print(expression, type(expression))
    # print(qubit_map)
    if isinstance(expression, Add):
        # Sum of multiple Pauli terms
        for term in expression.args:
            print(term, type(term))
        # print("Sum:", sum(pauli_term_measurement_expectation(term, frequencies, qubit_map) for term in expression.args))
        return sum(pauli_term_measurement_expectation(term, frequencies, qubit_map) for term in expression.args)
    # if not expression.args and isinstance(expression, (X, Y, Z)):
    elif isinstance(expression, (X, Y, Z)):
        z_only_ham = SymbolicHamiltonian(Z(expression.target_qubit), nqubits=expression.target_qubit + 1)
    elif isinstance(expression, Mul):
        # Single Pauli term
        pauli_z_terms = [Z(term.target_qubit) if isinstance(term, (X, Y, Z)) else term for term in expression.args]
        # pauli_z = [Z(int(factor.target_qubit)) for factor in pauli_term.factors if factor.name[0] != "I"]
        z_only_ham = SymbolicHamiltonian(reduce(lambda x, y: x * y, pauli_z_terms, 1.0))
    # Can now apply expectation_from_samples directly
    return z_only_ham.expectation_from_samples(frequencies, qubit_map=qubit_map)

    # z_terms = 0.0
    # Replace every (non-I) Symbol with Z, then include the term coefficient
    # for term, coefficient in hamiltonian.form.as_coefficients_dict().items():
    #     # if isinstance(term, One): # Return any constant term immediately
    #     #     z_terms += coefficient
    #     # else:
    #     if True:
    #         if isinstance(term, (X, Y, Z)):
    #             z_terms += coefficient * Z(term.target_qubit)
    #         else:
    #             pauli_z_terms = [
    #                 Z(factor.target_qubit) if isinstance(factor, (X, Y, Z)) else factor for factor in term.args
    #             ]
    #             z_terms += reduce(lambda x, y: x * y, pauli_z_terms, 1.0)

    # return SymbolicHamiltonian(z_terms).expectation_from_samples(frequencies, qubit_map=qubit_map)


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

    total = constant_term(hamiltonian)
    for _i, (expression, measurement_gates) in enumerate(grouped_terms):
        if measurement_gates:
            _circuit = circuit.copy()
            _circuit.add(measurement_gates)

            _circuit.draw()

            # Number of shots used to run the circuit depends on n_shots_per_pauli_term
            result = _circuit(nshots=n_shots if n_shots_per_pauli_term else shot_allocation[_i])

            frequencies = result.frequencies(binary=True)
            qubit_map = sorted(qubit for gate in measurement_gates for qubit in gate.target_qubits)
            if frequencies:  # Needed because might have cases whereby no shots allocated to a group
                # print("hamiltonian.form:", hamiltonian.form)
                # print(hamiltonian.form.as_coefficients_dict())
                # print("ham.form.args:", hamiltonian.form.args)
                # print("Expression:", expression, type(expression))
                total += pauli_term_measurement_expectation(expression, frequencies, qubit_map)
                # total += sum(
                #     pauli_term_measurement_expectation(expression, frequencies, qubit_map)
                #     for term, coefficient in hamiltonian.form.as_coefficients_dict().items()
                # )

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


# from qibo import Circuit, gates
#
# # hamiltonian = SymbolicHamiltonian(0.26 + 4.2 * Z(0) * X(2)) #  + Z(1) + 5 * X(0) * Y(1))
# hamiltonian = SymbolicHamiltonian(0.26 + 4.2 * Z(0) * X(2) + Z(1) + 5 * X(0) * Y(1))
#
# # result = measurement_basis_rotations(hamiltonian, "qwc")
# # for term in result:
# #     print(term, "M gates: ", [m_gate.target_qubits for m_gate in term[1]])
#
# n_qubits = hamiltonian.nqubits
# circuit = Circuit(n_qubits)
# circuit.add(gates.RX(_i, 0.13*_i+0.7**_i) for _i in range(n_qubits))
# # circuit.draw()
#
# exact = expectation(circuit, hamiltonian)
# print("Exact:", exact)
#
# from_samples = expectation_from_samples(circuit, hamiltonian, group_pauli_terms="qwc", n_shots=10000)
# print("From samples:", from_samples)
