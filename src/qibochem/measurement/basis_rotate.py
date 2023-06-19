import qibo
from qibo import gates
from qibo.models import Circuit


def measure_rotate_basis(pauli_op, nqubits):
    """
    rotate each qubit to computational basis (Z-basis) for measurement
    input:
        pauli_op: qubit operator
        nqubits: number of qubits
    output:
        qibo circuit with rotations and measurement gates
    """

    mqc = Circuit(nqubits)
    for p in pauli_op:
        if p[1] == "Z":
            mqc.add(gates.M(p[0]))
        elif p[1] == "Y":
            mqc.add(gates.S(p[0]).dagger())
            mqc.add(gates.H(p[0]))
            mqc.add(gates.M(p[0]))
        elif p[1] == "X":
            mqc.add(gates.H(p[0]))
            mqc.add(gates.M(p[0]))
    return mqc
