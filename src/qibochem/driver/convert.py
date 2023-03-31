import openfermion as of
from qibo import hamiltonians
from qibo.symbols import X, Y, Z, I

def ofqubitoperator_to_symham(of_qubit_op):
    """
    convert openfermion QubitOperator to qibo symbolic hamiltonian
    Args:
        of_qubit_op: OpenFermion QubitOperator
    Returns:
        Qibo symbolic hamiltonian
    """    
    ham = 0.0
    xyz_to_symbol = {'X': X, 'Y': Y, 'Z': Z}
    for term in of_qubit_op.terms:
        term_coeff = of_qubit_op.terms[term]
        if len(term) == 0:
            ham += of_qubit_op.terms[term]
        else:
            temp = term_coeff
            for local_op in term:
                temp *= xyz_to_symbol[local_op[1]](local_op[0])
                #print(local_op[1], local_op[0])
            ham += temp
        
    return hamiltonians.SymbolicHamiltonian(ham)
