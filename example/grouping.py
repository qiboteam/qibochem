"""
2020 - Efficient evaluation of quantum observables using entangled measurements

WARNING: Draft code that doesn't work!!!!

"""

import networkx as nx
import numpy as np
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import I, X, Y, Z

from qibochem.driver import Molecule
from qibochem.measurement import expectation
from qibochem.measurement.optimization import check_terms_commutativity

#
# ZC NOTE: Was trying to generate all combinations of qubit pairs. Seems like not supposed to do it this way...?
#
# def generate_entangled_measurements(measurements, entanglements=None):
#     """
#     Generate all possible pairs of qubits that can be measured using entangled measurements
#
#     Args:
#         measurements (dict): TPB measurements, e.g. {1: "X", 2: "Z", ...}
#         entanglement_type: Type of entangled measurements to consider. Currently only Bell measurements implemented
#
#     Returns:
#         (dict): Keys are qubits, values are the possible measurements (TPB and Bell) for that qubit
#     """
#     if entanglements is None:
#         entanglements = (("X", "X"), ("Y", "Y"), ("Z", "Z"))
#
#     result = {}
#     for _i, (qubit1, measurement1) in enumerate(measurements.items()):
#         remaining_qubits = list(measurements.keys())
#         # print(remaining_qubits)
#         new_measurements = {}
#         for _j, (qubit2, measurement2) in enumerate(measurements.items()):
#             if _j > _i:
#                 if (measurement1, measurement2) in entanglements:
#                     # Change the measurement to a Bell measurement if possible
#                     new_measurements = {**new_measurements, **{qubit1: "Bell", qubit2: "Bell"}}
#                     remaining_qubits.remove(qubit1)
#                     remaining_qubits.remove(qubit2)
#                     # print((qubit1, measurement1), (qubit2, measurement2))
#                     # print(remaining_qubits)
#                     break
#         # Add in the original TPB measurements for the remaining qubits.
#         new_measurements = {**new_measurements, **{qubit:measurements[qubit] for qubit in remaining_qubits}}
#         if new_measurements not in result:
#             result.append(new_measurements)
#         # print(result)
#         # print()
#     return result


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
                    if "Bell" not in result[qubit1]:
                        result[qubit1].append("Bell")
                    if "Bell" not in result[qubit2]:
                        result[qubit2].append("Bell")
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

    for _i, node1 in enumerate(all_nodes):
        if node1 in remaining_nodes:
            # Base case: grouping is empty, just add the first node in directly
            if not grouping:
                remaining_nodes.remove(node1)
                grouping.append([node1])
                measurement_groups.append(tpb_measurements[node1])
                # continue
            for _j, node2 in enumerate(all_nodes):
                if _j > _i and node2 in remaining_nodes:
                    # Should have something in grouping by now
                    group_index = None  # Flag to see which group to add node2 to
                    for _k, group in enumerate(grouping):

                        # Algorithm 2 in the reference paper. Not 100% sure is correct!
                        # Check whether current pair of Pauli strings is compatible
                        if check_compatible(all_measurements[node1], all_measurements[node2]):
                            print("Node1:", node1)
                            print("Node2:", node2)

                            # Get the measurements for the current group
                            measurements = measurement_groups[_k]
                            print(measurements)
                            # Generate permutations of overlapping qubit positions with different terms and check if
                            # any of the entangled measurements can be applied to the position
                            overlapping_qubits = [
                                qubit
                                for qubit, measurement in measurements.items()
                                if all_measurements[node2].get(qubit) is not None
                                and measurement in all_measurements[node2][qubit]
                            ]
                            # Try to replace single Pauli measurements with entangled measurements
                            new_measurements = {}
                            count = 0
                            while overlapping_qubits:
                                to_remove = []
                                print(f"Overlapping qubits: {overlapping_qubits}")
                                n_qubits = len(overlapping_qubits)
                                # Cannot have an odd number of overlapping qubits
                                if n_qubits % 2 == 1:
                                    new_measurements = {}
                                    break
                                for _l in range(n_qubits):
                                    qubit1 = overlapping_qubits[_l]
                                    for _m in range(n_qubits):
                                        qubit2 = overlapping_qubits[_m]
                                        if _m > _l and not to_remove:
                                            # Check to see if (qubit1, qubit2) can be measured using Bell measurements
                                            if (
                                                any(
                                                    (measurements[qubit1], m_type) in bell_measurements
                                                    for m_type in all_measurements[node2][qubit2]
                                                )
                                                or (all_measurements[node2][qubit1], all_measurements[node2][qubit1])
                                                in bell_measurements
                                            ):
                                                new_measurements = {
                                                    **new_measurements,
                                                    **{qubit1: "Bell", qubit2: "Bell"},
                                                }
                                                to_remove.append(qubit1)
                                                to_remove.append(qubit2)
                                                break
                                for qubit in to_remove:
                                    overlapping_qubits.remove(qubit)
                                count += 1
                                if count > 100:
                                    break
                            # Any successful change in the measurements?
                            if new_measurements:
                                group_index = _k
                                # Update measurements dictionary
                                measurements = {**measurements, **new_measurements}
                                # Add measurements for the remaining qubits (in node2 but not in node1)
                                measurements = {
                                    **measurements,
                                    **{
                                        qubit: measurement
                                        for qubit, measurement in tpb_measurements[node2].items()
                                        if qubit not in measurements.keys()
                                    },
                                }

                    if group_index is None:
                        grouping.append([node2])
                        measurement_groups.append(tpb_measurements[node2])
                    else:
                        grouping[group_index].append(node2)
                        measurement_groups[group_index] = measurements
                    remaining_nodes.remove(node2)

                print("Remaining nodes:")
                print(remaining_nodes)

    print("\nFinal grouping")
    print(grouping)

    # Check to ensure all nodes were placed in a group
    grouped_terms = {term for group in grouping for term in group}
    assert grouped_terms == set(all_nodes)


if __name__ == "__main__":
    main()
