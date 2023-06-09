"""
Sample script of how to build and run the UCCSD ansatz with HF embedding for LiH
"""

import numpy as np

from scipy.optimize import minimize
from qibo.optimizers import optimize

from qibochem.driver.molecule import Molecule

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.ucc import ucc_circuit

# Define molecule and populate
mol = Molecule(xyz_file="lih.xyz")
try:
    mol.run_pyscf()
except ModuleNotFoundError:
    mol.run_psi4()


# Apply embedding and boson encoding
mol.hf_embedding(active=[1, 2, 5])
hamiltonian = mol.hamiltonian(
    oei=mol.embed_oei, tei=mol.embed_tei, constant=mol.inactive_energy
)

# Set parameters for the rest of the experiment
n_qubits = mol.n_active_orbs
n_electrons = mol.n_active_e

# Build circuit
circuit = hf_circuit(n_qubits, n_electrons)

# UCCSD: Excitations
d_excitations = [(_i, _j, _a, _b)
                 for _i in range(n_electrons) for _j in range(_i+1, n_electrons) # Electrons
                 for _a in range(n_electrons, n_qubits) for _b in range(_a+1, n_qubits) # Orbs
                 if (_i + _j + _a + _b) % 2 == 0 and ((_i%2 + _j%2) == (_a%2 + _b%2)) # Spin
                ]
s_excitations = [(_i, _a,) for _i in range(n_electrons) for _a in range(n_electrons, n_qubits)
                 if (_i + _a) % 2 == 0 # Spin-conservation
                ]
# Sort excitations with very contrived lambda functions
d_excitations = sorted(d_excitations, key=lambda x: (x[3] - x[2]) + (x[2]%2))
s_excitations = sorted(s_excitations, key=lambda x: (x[1] - x[0]) + (x[0]%2))
excitations = d_excitations + s_excitations
n_excitations = len(excitations)
# print(excitations)
# Output: [(0, 1, 2, 3), (0, 1, 4, 5), (0, 1, 3, 4), (0, 1, 2, 5), (0, 2), (1, 3), (0, 4), (1, 5)]
# Only 5 distinct excitations...
n_unique_excitations = 5

# UCCSD: Circuit
all_coeffs = []
for _ex in excitations:
    coeffs = []
    circuit += ucc_circuit(n_qubits, 0.0, _ex, coeffs=coeffs)
    all_coeffs.append(coeffs)

# Draw the circuit if interested
print(circuit.draw())
print()


def electronic_energy(parameters):
    r"""
    Loss function for the UCCSD ansatz
    """
    all_parameters = []

    # UCC parameters
    # Expand the parameters to match the total UCC ansatz manually
    _ucc = parameters[:n_unique_excitations]
    # Manually group the related excitations together
    ucc_parameters = [_ucc[0], _ucc[1], _ucc[2], _ucc[2], _ucc[3], _ucc[3], _ucc[4], _ucc[4]]
    # Need to iterate through each excitation this time
    for _coeffs, _parameter in zip(all_coeffs, ucc_parameters):
        # Convert a single value to a array with dimension=n_param_gates
        ucc_parameter = np.repeat(_parameter, len(_coeffs))
        # Multiply by coeffs
        ucc_parameter *= _coeffs
        all_parameters.append(ucc_parameter)

    # Flatten all_parameters into a single list to set the circuit parameters
    all_parameters = [_x for _param in all_parameters for _x in _param]
    circuit.set_parameters(all_parameters)

    return mol.expectation(circuit, hamiltonian)


# Reference energy
fci_energy = hamiltonian.eigenvalues()[0]

# Random initialization
params = np.random.rand(n_unique_excitations)

best, params, extra = optimize(electronic_energy, params)

print("\nResults using Qibo optimize:")
print(f"FCI energy: {fci_energy:.8f}")
print(f" HF energy: {mol.e_hf:.8f} (Classical)")
print(f"VQE energy: {best:.8f} (UCCSD ansatz)")
# print()
# print("Optimized parameters:", params)


# Scipy minimize
params = np.random.rand(n_unique_excitations)

result = minimize(electronic_energy, params)
best, params = result.fun, result.x

print("\nResults using scipy.optimize:")
print(f"FCI energy: {fci_energy:.8f}")
print(f" HF energy: {mol.e_hf:.8f} (Classical)")
print(f"VQE energy: {best:.8f} (UCCSD ansatz)")
# print()
# print("Optimized parameters:", params)


full_ham = mol.hamiltonian("f")
mol_fci_energy = mol.eigenvalues(full_ham)[0]

print(f"\nFCI energy: {mol_fci_energy:.8f} (Full Hamiltonian)")
