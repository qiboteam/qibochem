from qibo import gates


def hea(nlayers, nqubits, parameter_gates=["RY", "RZ"], coupling_gates="CZ"):
    gate_list = []

    for ilayer in range(nlayers):
        for rgate in parameter_gates:
            for iqubit in range(nqubits):
                gate_list.append(getattr(gates, rgate)(iqubit, theta=0))

        # entanglement
        for iqubit in range(nqubits - 1):
            gate_list.append(getattr(gates, coupling_gates)(iqubit, iqubit + 1))
        gate_list.append(getattr(gates, coupling_gates)(nqubits - 1, 0))

    return gate_list
