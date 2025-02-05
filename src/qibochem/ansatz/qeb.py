from qibo import Circuit, gates


def qeb_circuit(n_qubits, excitation, theta=0.0) -> Circuit:
    r"""
    Qubit-excitation-based (QEB) circuit corresponding to the unitary coupled-cluster ansatz for a single excitation

    Supports only Jordan-Wigner encoded circuits

    Ref: arXiv:2210.05771

    Args:
        n_qubits: Number of qubits in the quantum circuit
        excitation: Iterable of orbitals involved in the excitation; must have an even number of elements
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``
        theta: UCC parameter. Defaults to 0.0
        trotter_steps: Number of Trotter steps; i.e. number of times this ansatz is applied with ``theta`` = ``theta / trotter_steps``. Default: 1

    Returns:
        Qibo ``Circuit``: Circuit corresponding to a single UCC excitation
    """

    n_orbitals = len(excitation)
    assert n_orbitals % 2 == 0, f"{excitation} must have an even number of items"

    n_tuples = len(excitation) // 2
    i_array = excitation[:n_tuples]
    a_array = excitation[n_tuples:]
    fwd_gates = [gates.CNOT(i_array[-1], _i) for _i in i_array[-2::-1]]
    fwd_gates += [gates.CNOT(a_array[-1], _a) for _a in a_array[-2::-1]]
    fwd_gates.append(gates.CNOT(a_array[-1], i_array[-1]))
    fwd_gates += [gates.X(_ia) for _ia in excitation if _ia not in (i_array[-1], a_array[-1])]
    circuit = Circuit(n_qubits)
    circuit.add(gate for gate in fwd_gates)
    # MCRY
    # multi-controlled RY gate,
    # negative controls i, a
    # positive control on i_n
    mcry_controls = excitation[:-1]
    ry_angle = 2.0 * theta
    circuit.add(gates.RY(a_array[-1], ry_angle).controlled_by(*mcry_controls))
    circuit.add(gate for gate in fwd_gates[::-1])
    return circuit
