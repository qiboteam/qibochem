from qibo import gates


def hea_su2(nlayers, nqubits):
    """
    implements a hardware-efficient ansatz SU2
    inputs:
    """
    gate_list = []

    for ilayer in range(nlayers):
        for iqubit in range(nqubits):
            gate_list.append(gates.RY(iqubit, theta=0))
        for iqubit in range(nqubits):
            gate_list.append(gates.RZ(iqubit, theta=0))
        # entanglement
        for iqubit in range(nqubits - 1):
            gate_list.append(gates.CNOT(iqubit, iqubit + 1))
        gate_list.append(gates.CNOT(nqubits - 1, 0))

    for iqubit in range(nqubits):
        gate_list.append(gates.RY(iqubit, theta=0))
    for iqubit in range(nqubits):
        gate_list.append(gates.RZ(iqubit, theta=0))

    return gate_list
