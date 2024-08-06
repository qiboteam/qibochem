import numpy as np 
import qibo
from qibo import Circuit, models, gates

def MCRY(controls, target, parameter, trainable=True) -> Circuit:
    """
    multi-controlled RY gate
    """
    gate_list = []

    if len(controls) > 2:
        gate_list.append(gates.CRY(controls[-1], target, parameter/2.0, trainable=trainable))        
        gate_list.append(gates.CRY(controls[-1], target, parameter/2.0, trainable=trainable))
        gate_list.append(gates.X(controls[-1]).controlled_by(*controls[0:(len(controls)-1)]))
        gate_list.append(gates.CRY(controls[-1], target, -parameter/2.0, trainable=trainable))
        gate_list.append(gates.X(controls[-1]).controlled_by(*controls[0:(len(controls)-1)]))
        _temp = MCRY(controls[0:(len(controls)-1)], target, parameter/2.0)
        gate_list += _temp
    else:
        gate_list.append(gates.CRY(controls[1], target, parameter/2.0, trainable=trainable))
        gate_list.append(gates.CNOT(controls[0], controls[1]))
        gate_list.append(gates.CRY(controls[1], target, -parameter/2.0, trainable=trainable))
        gate_list.append(gates.CNOT(controls[0], controls[1]))
        gate_list.append(gates.CRY(controls[0], target, parameter/2.0, trainable=trainable))

    return gate_list

def FEB_circuit(n_qubits, excitation, theta=0.0, trotter_steps=1, decompose=True) -> Circuit:

    n_orbitals = len(excitation)
    assert n_orbitals % 2 == 0, f"{excitation} must have an even number of items"

    n_tuples = len(excitation) // 2
    i_array = excitation[0:n_tuples]
    a_array = excitation[n_tuples:]

    ry_angle = 2.0*theta
    if n_tuples % 4 in [2,3]: 
        ry_angle *= -1

    fwd_list = []

    for _c in reversed(range(i_array[0]+2, a_array[-1])):
        fwd_list.append(gates.CNOT(_c, _c-1))
        
    if len(excitation) > 2:
        fwd_list.append(gates.CZ(a_array[-1], i_array[0]+1))
                    
    for _a in reversed(a_array[0:-1]):
        fwd_list.append(gates.CNOT(a_array[-1], _a))
    
    for _i in reversed(i_array[0:-1]):
        fwd_list.append(gates.CNOT(i_array[-1], _i))
    
    fwd_list.append(gates.CNOT(a_array[-1], i_array[-1]))
    
    for _ia in excitation:
        if _ia not in [i_array[-1], a_array[-1]]:
            fwd_list.append(gates.X(_ia))

    if len(excitation) > 2:
        mcry_gate = MCRY(excitation[:-1], excitation[-1], ry_angle)
    else:
        mcry_gate = [gates.CRY(i_array[0], a_array[0], ry_angle)]

    gate_list = []

    for _g in fwd_list:
        gate_list.append(_g)
        
    for _g in mcry_gate: 
        gate_list.append(_g)

    for _g in reversed(fwd_list):
        gate_list.append(_g)    

    circuit = Circuit(n_qubits)
    if decompose is True:
        for _g in gate_list:
            controls = list(_g.control_qubits)
            targets = list(_g.target_qubits)
            qubit_span = controls + targets
            qubits = list(range(n_qubits))
            free_qubits = list(set(qubits) - set(qubit_span))
            if len(free_qubits) == 0:
                free_qubits = [excitation[-1]]
            if len(controls) > 2:
                mcx_decomposition = _g.decompose(*free_qubits)
                circuit.add(mcx_decomposition)
            else:
                circuit.add(_g)
    else:
        circuit.add(gate_list)

    return circuit



