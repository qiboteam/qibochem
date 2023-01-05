from qibo import gates
from qibo.models import Circuit

def hea_su2(nlayers, nqubits):
    '''
    implements a hardware-efficient ansatz SU2
    inputs:
    '''
    qc = Circuit(nqubits)
    
    for ilayer in range(nlayers):
        for iqubit in range(nqubits):
            qc.add(gates.RY(iqubit, theta=0))
        for iqubit in range(nqubits):
            qc.add(gates.RZ(iqubit, theta=0))
        #entanglement
        for iqubit in range(nqubits-1):
            qc.add(gates.CNOT(iqubit, iqubit+1))
        qc.add(gates.CNOT(nqubits-1, 0))
        
    for iqubit in range(nqubits):
        qc.add(gates.RY(iqubit, theta=0))
    for iqubit in range(nqubits):
        qc.add(gates.RZ(iqubit, theta=0))
        
    return qc
