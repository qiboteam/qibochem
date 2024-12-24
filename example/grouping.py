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


def check_compatible(term1_measurements, term2_measurements):
    """Must have common measurements for the qubits present in both terms"""
    compatible = True
    overlapping_qubits = set(term1_measurements.keys()) & set(term2_measurements.keys())
    if overlapping_qubits:
        compatible = all(
            set(term1_measurements[qubit]) & set(term2_measurements[qubit]) for qubit in overlapping_qubits
        )
    return compatible


def select_measurement(possible_measurements):
    """
    Select appropriate measurements from all possible measurements. Roughly corresponds to lines 6 to 17 in Algorithm 2
    TODO: Add other entangled measurements beside Bell measurements
    """
    # Add in single qubit Pauli measurements first
    result = {
        qubit: measurements[0]
        for qubit, measurements in possible_measurements.items()
        if len(measurements) == 1 and len(measurements[0]) == 1
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
    mol = Molecule(xyz_file="../h2.xyz")
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
    grouping = []

    print("\nAll nodes:")
    all_nodes = [node for node, degree in sorted_nodes]
    print(all_nodes)
    remaining_nodes = list(all_nodes)  # Make a copy of all_nodes to track which have been merged
    measurement_groups = []  # Also need to record what measurements to use for each group

    while remaining_nodes:
        for _i, node1 in enumerate(all_nodes):
            if node1 in remaining_nodes:
                print("Node1:", node1)
                # Base case: grouping is empty, just add the first node in directly
                if not grouping:
                    remaining_nodes.remove(node1)
                    grouping.append([node1])
                    measurement_groups.append(tpb_measurements[node1])
                    # continue
                for _j, node2 in enumerate(all_nodes):
                    if _j > _i and node2 in remaining_nodes:
                        print("Node2:", node2)
                        # Should have something in grouping by now
                        group_index = None  # Flag to see which group to add node2 to
                        for _k, group in enumerate(grouping):
                            print(f"Group {_k}: {group}")
                            # Get the measurements for the current group
                            current_measurements = measurement_groups[_k]
                            # Add entangled measurements in as well
                            measurements = generate_entangled_measurements(current_measurements)
                            print(f"Current possible measurements: {measurements}")

                            # Algorithm 2 in the reference paper. Not 100% sure is correct!
                            # Check whether current pair of Pauli strings is compatible with current set of measurements
                            if check_compatible(measurements, all_measurements[node2]):
                                print(f"node1 ({measurements}) and node2 ({all_measurements[node2]}) compatible!")
                                # Generate permutations of overlapping qubit positions with different terms and check if
                                # any of the entangled measurements can be applied to the position (???)
                                possible_new_measurements = {
                                    qubit: sorted(possible_measurements, key=len)
                                    for qubit in set(measurements.keys()) & set(all_measurements[node2].keys())
                                    if (
                                        possible_measurements := set(measurements[qubit])
                                        & set(all_measurements[node2][qubit])
                                    )
                                }
                                # Add on measurements for qubits in current set of measurements, but not in the new node considered
                                possible_new_measurements = {
                                    **measurements,
                                    **possible_new_measurements,
                                }
                                print(f"Possible new measurements: {possible_new_measurements} (Overlap with current)")
                                if not possible_new_measurements:
                                    possible_new_measurements = {**measurements, **all_measurements[node2]}
                                    print(
                                        f"Possible new measurements: {possible_new_measurements} (No overlap with current)"
                                    )
                                # Add on measurements for qubits not in current set of measurements, but in the new node considered
                                # TODO: Probably can merge with the previous if condition...?
                                possible_new_measurements = {**all_measurements[node2], **possible_new_measurements}

                                new_measurements = select_measurement(possible_new_measurements)
                                print(f"New measurements: {new_measurements}")
                                # Add measurements for the remaining qubits (in node2 but not in current set of measurements)
                                new_measurements = {
                                    **new_measurements,
                                    **{
                                        qubit: measurement
                                        for qubit, measurement in tpb_measurements[node2].items()
                                        if qubit not in new_measurements.keys()
                                    },
                                }

                                group_index = _k
                                break
                            else:
                                print(f"node1 ({node1}) and node2 ({node2}) not compatible!")

                        if group_index is None:
                            print("No current group compatible. Creating new measurement group")
                            grouping.append([node2])
                            measurement_groups.append(tpb_measurements[node2])
                        else:
                            print(f"Adding {new_measurements} to group {group_index}")
                            grouping[group_index].append(node2)
                            measurement_groups[group_index] = new_measurements
                        remaining_nodes.remove(node2)

                        print()

                    print("Remaining nodes:")
                    print(remaining_nodes)
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
    main()
