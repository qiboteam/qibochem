"""
Sample script of how to build and run the UCCSD ansatz with HF embedding for LiH
"""

import numpy as np
from qibo.optimizers import optimize
from scipy.optimize import minimize

from qibochem.ansatz import hf_circuit, ucc_circuit
from qibochem.driver import Molecule
from qibochem.measurement import expectation

# Define molecule and populate
mol = Molecule(xyz_file="lih.xyz")
mol.run_pyscf()


# Apply embedding and boson encoding
mol.hf_embedding(active=[1, 2, 5])
hamiltonian = mol.hamiltonian()

# Set parameters for the rest of the experiment
n_qubits = mol.n_active_orbs
n_electrons = mol.n_active_e

# Build circuit
circuit = hf_circuit(n_qubits, n_electrons)

# UCCSD: Excitations
d_excitations = [
    (_i, _j, _a, _b)
    for _i in range(n_electrons)
    for _j in range(_i + 1, n_electrons)  # Electrons
    for _a in range(n_electrons, n_qubits)
    for _b in range(_a + 1, n_qubits)  # Orbs
    if (_i + _j + _a + _b) % 2 == 0 and ((_i % 2 + _j % 2) == (_a % 2 + _b % 2))  # Spin
]
s_excitations = [
    (_i, _a)
    for _i in range(n_electrons)
    for _a in range(n_electrons, n_qubits)
    if (_i + _a) % 2 == 0  # Spin-conservation
]
# Sort excitations with very contrived lambda functions
d_excitations = sorted(d_excitations, key=lambda x: (x[3] - x[2]) + (x[2] % 2))
s_excitations = sorted(s_excitations, key=lambda x: (x[1] - x[0]) + (x[0] % 2))
excitations = d_excitations + s_excitations
n_excitations = len(excitations)
# print(excitations)
# Output: [(0, 1, 2, 3), (0, 1, 4, 5), (0, 1, 3, 4), (0, 1, 2, 5), (0, 2), (1, 3), (0, 4), (1, 5)]
# Only 5 distinct excitations...
n_unique_excitations = 5

# UCCSD: Circuit
for _ex in excitations:
    circuit += ucc_circuit(n_qubits, _ex)

# Draw the circuit if interested
print(circuit.draw())
print()


def electronic_energy(parameters):
    r"""
    Loss function for the UCCSD ansatz
    """
    coeff_dict = {1: (-1.0, 1.0), 2: (-0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25)}

    # Unique UCC parameters
    # Manually group the related excitations together
    ucc_parameters = [
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[2],
        parameters[3],
        parameters[3],
        parameters[4],
        parameters[4],
    ]
    all_parameters = []
    # Iterate through each excitation, with the list of coefficients dependent on whether S/D excitation
    for _ex, _parameter in zip(excitations, ucc_parameters):
        coeffs = coeff_dict[len(_ex) // 2]
        # Convert a single value to a array with dimension=n_param_gates
        ucc_parameter = np.repeat(_parameter, len(coeffs))
        # Multiply by coeffs
        ucc_parameter *= coeffs
        all_parameters.append(ucc_parameter)

    # Flatten all_parameters into a single list to set the circuit parameters
    all_parameters = [_x for _param in all_parameters for _x in _param]
    circuit.set_parameters(all_parameters)

    return expectation(circuit, hamiltonian)


# Reference energy
fci_energy = hamiltonian.eigenvalues()[0]

# Random initialization
params = np.random.rand(n_unique_excitations)

best, params, extra = optimize(electronic_energy, params)

print("\nResults using Qibo optimize: (With HF embedding)")
print(f"FCI energy: {fci_energy:.8f}")
print(f" HF energy: {mol.e_hf:.8f} (Classical)")
print(f"VQE energy: {best:.8f} (UCCSD ansatz)")
# print()
# print("Optimized parameters:", params)


# Scipy minimize
params = np.random.rand(n_unique_excitations)

result = minimize(electronic_energy, params)
best, params = result.fun, result.x

print("\nResults using scipy.optimize: (With HF embedding)")
print(f"FCI energy: {fci_energy:.8f}")
print(f" HF energy: {mol.e_hf:.8f} (Classical)")
print(f"VQE energy: {best:.8f} (UCCSD ansatz)")
# print()
# print("Optimized parameters:", params)


full_ham = mol.hamiltonian("f", oei=mol.oei, tei=mol.tei, constant=0.0)
mol_fci_energy = mol.eigenvalues(full_ham)[0]

print(f"\nFCI energy: {mol_fci_energy:.8f} (Full Hamiltonian)")
