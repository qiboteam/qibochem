"""
2020 - Efficient evaluation of quantum observables using entangled measurements

WARNING: Draft code that doesn't work!!!!

Bug: Check of compatibility between one-qubit measurements (e.g. "Z0") with Bell measurements (e.g. "Bell01) fails;
- Should be incompatible, but currently returning compatible

Otherwise seems to be OK...?
"""

import networkx as nx
import numpy as np
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import I, X, Y, Z

from qibochem.driver import Molecule
from qibochem.measurement import expectation
from qibochem.measurement.optimization import check_terms_commutativity


def generate_entangled_measurements(measurements, entanglements=None):
    """
    Generate the set of possible measurements for each qubit. Only considering TPB and Bell at the moment

    Args:
        measurements (dict): TPB measurements, e.g. {1: "X", 2: "Z", ...}
        entanglement_type: Type of entangled measurements to consider. Currently only Bell measurements implemented

    Returns:
        (dict): Keys are qubits, values are the possible measurements (TPB and Bell) for that qubit
    """
    if entanglements is None:
        entanglements = (("X", "X"), ("Y", "Y"), ("Z", "Z"))

    result = {
        qubit: [
            measurement,
        ]
        for qubit, measurement in measurements.items()
    }
    for _i, (qubit1, measurement1) in enumerate(measurements.items()):
        for _j, (qubit2, measurement2) in enumerate(measurements.items()):
            if _j > _i:
                if (measurement1, measurement2) in entanglements:
                    # Change the measurement to a Bell measurement if possible
                    if f"Bell{qubit1}{qubit2}" not in result[qubit1]:
                        result[qubit1].append(f"Bell{qubit1}{qubit2}")
                    if f"Bell{qubit1}{qubit2}" not in result[qubit2]:
                        result[qubit2].append(f"Bell{qubit1}{qubit2}")
    return result


def disallowed_entanglements(measurements):
    """
    Finds the qubit which cannot be measured using entangled measurements. For odd numbers of measurements, the last
    qubit has to be a single qubit Pauli measurement. If >1 qubit with Pauli measurements, still can be flexibile
    """
    result = set()
    pauli_measurements = [qubit for qubit, measurement in measurements.items() if len(measurement) == 1]
    if len(pauli_measurements) == 1:
        result.add(pauli_measurements[0])
    return result


def check_compatible(term1_measurements, term2_measurements):
    """Must have common measurements for the qubits present in both terms"""
    compatible = True
    overlapping_qubits = set(term1_measurements.keys()) & set(term2_measurements.keys())
    if overlapping_qubits:
        compatible = all(
            set(term1_measurements[qubit]) & set(term2_measurements[qubit]) for qubit in overlapping_qubits
        )
    return compatible


def select_measurement(possible_measurements, no_entanglements):
    """
    Select appropriate measurements from all possible measurements. Roughly corresponds to lines 6 to 17 in Algorithm 2
    TODO: Add other entangled measurements beside Bell measurements

    Args:
        possible_measurements (dict): (qubit, list of possible measurements)
        no_entanglements (list): Qubits that have to be single qubit Pauli measurements
    """
    # Add in single qubit Pauli measurements first
    result = {
        qubit: measurements[0]
        for qubit, measurements in possible_measurements.items()
        if len(measurements[0]) == 1 and (len(measurements) == 1 or qubit in no_entanglements)
    }
    # Add possible Bell measurements next
    n_entangled_measurements = 0
    to_remove = None
    remaining_qubits = list(qubit for qubit in possible_measurements.keys() if qubit not in result.keys())
    while remaining_qubits:
        # Find the possible Bell pairs of qubits and select the first pair
        # TODO: Could probably write this more nicely
        bell_pair = None
        for qubit in remaining_qubits:
            for measurement in possible_measurements[qubit]:
                if measurement.startswith("Bell"):
                    possible_bell_pair = (int(measurement[4]), int(measurement[5]))
                    if all(_q in remaining_qubits for _q in possible_bell_pair):
                        bell_pair = possible_bell_pair
                    break
            if bell_pair is not None:
                break
        # print("Bell pair:", bell_pair)

        if bell_pair is None:
            print("Cannot find any pair of qubits to use entangled measurements!")
            # Set all qubits to be single Pauli measurements
            result = {
                **result,
                **{
                    qubit: possible_measurements[qubit][0]  # Should be sorted to give single qubit Pauli terms first
                    for qubit in remaining_qubits
                },
            }
            break
        for qubit in bell_pair:
            result[qubit] = f"Bell{bell_pair[0]}{bell_pair[1]}"
            remaining_qubits.remove(qubit)
    return result


def main():
    # Global variables:
    bell_measurements = (("X", "X"), ("Y", "Y"), ("Z", "Z"))

    # Define model Hamiltonians
    # mol = Molecule(xyz_file="../h2.xyz")
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74804))])
    # mol = Molecule(xyz_file="../lih.xyz")
    mol.run_pyscf()
    # mol.hf_embedding(active=[0, 1, 2, 3, 4])
    full_hamiltonian = mol.hamiltonian(ferm_qubit_map="bk")
    # full_hamiltonian = mol.hamiltonian()

    # for term in full_hamiltonian.terms:
    #     print(term.coefficient, term.factors)

    ham_terms = [" ".join(factor.name for factor in term.factors) for term in full_hamiltonian.terms]
    print(ham_terms)

    # Build Pauli graph
    G = nx.Graph()
    G.add_nodes_from(ham_terms)

    # Add an edge between nodes if they DON'T commute qubitwise
    G.add_edges_from(
        (term1, term2)
        for _i1, term1 in enumerate(ham_terms)
        for _i2, term2 in enumerate(ham_terms)
        if _i2 > _i1 and not check_terms_commutativity(term1, term2, qubitwise=True)
    )
    print("\nEdges")
    print(G.edges)

    # Sort the nodes by their degree
    print("\nDegree of nodes")
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    print(sorted_nodes)

    # Initialise assignment of measurements for all nodes
    tpb_measurements = {}
    for node, _degree in sorted_nodes:
        # Convert string representation of node into dictionary of measurements
        measurements = {int(factor[1:]): factor[0] for factor in node.split()}
        tpb_measurements[node] = measurements
    print("\nTPB measurements")
    print(tpb_measurements)

    # Search for the possible entangled measurements for each term
    all_measurements = {
        node: generate_entangled_measurements(measurements) for node, measurements in tpb_measurements.items()
    }
    # print("\nPossible measurements for each term")
    # for node, measurements in all_measurements.items():
    #     print(node)
    #     print(measurements)
    #     print()
    # return

    # Test check_compatible
    node1 = sorted_nodes[0][0]
    node2 = sorted_nodes[1][0]
    print(f"\n{node1} and {node2} compatible? {check_compatible(all_measurements[node1], all_measurements[node2])}")

    # Iterate over all nodes to merge compatible nodes
    # In practice: Define the new grouping as a list of lists.
    grouping = []  # Grouping of Hamiltonian terms
    measurement_groups = []  # Also need to record what measurements to use for each group
    disallowed_entanglement_groups = []  # Record qubits which cannot be used for Bell measurements

    print("\nAll nodes:")
    all_nodes = [node for node, degree in sorted_nodes]
    print(all_nodes)
    print()
    remaining_nodes = list(all_nodes)  # Make a copy of all_nodes to track which have been merged

    while remaining_nodes:
        group_index = None  # Flag to see which group to add node to
        node = remaining_nodes[0]
        print("Node:", node)
        node_no_entanglements = disallowed_entanglements(tpb_measurements[node])
        # If something in grouping, loop through each group and check compatibility of current node
        if grouping:
            for _k, group in enumerate(grouping):
                print(f"Group {_k}: {group}")
                # Get the measurements for the current group
                current_measurements = measurement_groups[_k]
                # Add entangled measurements in as well
                measurements = generate_entangled_measurements(current_measurements)
                print(f"Current possible measurements: {measurements}")
                # Combine the set of forbidden entanglements for the current node and group
                no_entanglements = disallowed_entanglement_groups[_k] | node_no_entanglements
                print(f"Disallowed entanglements: {no_entanglements}")

                # Algorithm 2 in the reference paper. Not 100% sure is correct!
                # Check whether current pair of Pauli strings is compatible with current set of measurements
                if check_compatible(measurements, all_measurements[node]):
                    print(f"Measurement group ({measurements}) compatible with current node ({all_measurements[node]})")
                    # Generate permutations of overlapping qubit positions with different terms and check if
                    # any of the entangled measurements can be applied to the position (???)
                    possible_new_measurements = {
                        qubit: sorted(possible_measurements, key=len)
                        for qubit in set(measurements.keys()) & set(all_measurements[node].keys())
                        if (possible_measurements := set(measurements[qubit]) & set(all_measurements[node][qubit]))
                    }
                    # Add on measurements for qubits in current set of measurements, but not in the new node considered
                    possible_new_measurements = {
                        **measurements,
                        **possible_new_measurements,
                    }
                    print(f"Possible new measurements: {possible_new_measurements} (Overlap with current)")
                    if not possible_new_measurements:
                        possible_new_measurements = {**measurements, **all_measurements[node]}
                        print(f"Possible new measurements: {possible_new_measurements} (No overlap with current)")
                    new_measurements = select_measurement(possible_new_measurements, no_entanglements)

                    # Add measurements for qubits in current node but not in current set of measurements
                    new_measurements = {
                        **new_measurements,
                        **{
                            qubit: measurement
                            for qubit, measurement in tpb_measurements[node].items()
                            if qubit not in new_measurements.keys()
                        },
                    }
                    print(f"New measurements: {new_measurements}")

                    group_index = _k
                    break
                else:
                    print(
                        f"Possible measurements for current node ({all_measurements[node]}) not compatible with "
                        f"current group ({measurements})"
                    )

        # Add current node to a group and remove it from remaining_nodes
        if group_index is None:
            print("No current group compatible. Creating new measurement group")
            grouping.append([node])
            measurement_groups.append(tpb_measurements[node])
            disallowed_entanglement_groups.append(node_no_entanglements)
        else:
            grouping[group_index].append(node)
            measurement_groups[group_index] = new_measurements
            no_entanglements = disallowed_entanglements(new_measurements)
            disallowed_entanglement_groups[group_index] |= no_entanglements
            print(
                f"Adding {node} to group {group_index}, new_measurements: {new_measurements}, "
                f"no_entanglements: {disallowed_entanglement_groups[group_index]}"
            )
        remaining_nodes.remove(node)
        print()

    # Check to ensure all nodes were placed in a group
    grouped_terms = {term for group in grouping for term in group}
    assert grouped_terms == set(all_nodes)

    print("\nFinal grouping and measurements")
    for group, measurement in zip(grouping, measurement_groups):
        print(group)
        print(measurement)
        print()


if __name__ == "__main__":
    # Hopefully correct result for H2/BK mapping: Two groups
    #
    # Group 1:
    # Terms: ['X0 Z1 X2', 'Y0 Z1 Y2', 'X0 Z1 X2 Z3', 'Y0 Z1 Y2 Z3', 'Z0 Z2', 'Z0 Z1 Z2', 'Z0 Z2 Z3', 'Z0 Z1 Z2 Z3', 'Z1', 'Z1 Z3']
    # Measurements: {0: 'Bell02', 1: 'Z', 2: 'Bell02', 3: 'Z'}
    #
    # Group 2:
    # Terms: ['Z0', 'Z2', 'Z0 Z1', 'Z1 Z2 Z3']
    # Measurements: {0: 'Z', 1: 'Z', 2: 'Z', 3: 'Z'}
    main()
