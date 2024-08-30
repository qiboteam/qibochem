import numpy as np 
import qibo
from qibo import Circuit, models, gates

def qeb_circuit(n_qubits, excitation, theta=0.0, trotter_steps=1) -> Circuit:
    
    n_orbitals = len(excitation)
    assert n_orbitals % 2 == 0, f"{excitation} must have an even number of items"

    n_tuples = len(excitation) // 2
    i_array = excitation[0:n_tuples]
    a_array = excitation[n_tuples:]

    ry_angle = 2.0*theta

    mcry_controls = excitation[0:-1]
    
    fwd_gates = []

    for _i in reversed(i_array[0:-1]):
        fwd_gates.append(gates.CNOT(i_array[-1], _i))
    
    for _a in reversed(a_array[0:-1]):
        fwd_gates.append(gates.CNOT(a_array[-1], _a))
    
    fwd_gates.append(gates.CNOT(a_array[-1], i_array[-1]))

    # MCRY
    # multi-controlled RY gate,
    # negative controls i, a
    # positive control on i_n
    for _ia in excitation:
        if _ia not in [i_array[-1], a_array[-1]]:
            fwd_gates.append(gates.X(_ia))

    mcry_gate = []
    mcry_gate.append(gates.RY(a_array[-1], ry_angle).controlled_by(*mcry_controls))
    
    gate_list = []

    for _g in fwd_gates:
        gate_list.append(_g)
    for _g in mcry_gate:
        gate_list.append(_g)
    for _g in reversed(fwd_gates):
        gate_list.append(_g)

    
    circuit = Circuit(n_qubits)
    for _g in gate_list:
        circuit.add(_g)
    
    return circuit    



