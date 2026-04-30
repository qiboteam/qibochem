"""
Sample script of how to build and run the UCCSD ansatz with HF embedding for LiH
"""

import numpy as np
from qibo.optimizers import optimize

from qibochem.ansatz import generate_excitations, sort_excitations, ucc_ansatz
from qibochem.driver import Molecule


def electronic_energy(parameters, hamiltonian, circuit, excitations):
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

    return hamiltonian.expectation(circuit)


def main():
    """Main function"""
    # Define molecule with HF embedding and get the molecular Hamiltonian
    mol = Molecule(xyz_file="lih.xyz")
    mol.run_pyscf()
    mol.hf_embedding(active=[1, 2, 5])
    hamiltonian = mol.hamiltonian()

    # Set parameters for the rest of the experiment
    n_qubits = mol.n_active_orbs
    n_electrons = mol.n_active_e

    # UCCSD: Get the list of excitations for constructing the circuit ansatz
    excitations = []
    for order in range(2, 0, -1):  # Reversed to get higher excitations first
        excitations += sort_excitations(
            generate_excitations(order, range(0, n_electrons), range(n_electrons, n_qubits))
        )
    print(f"Excitations: {excitations}\n")
    # Output: [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 5], [0, 1, 3, 4], [0, 2], [1, 3], (0, 4), (1, 5)]
    # Note that there are strictly only 5 distinct excitations w.r.t. the MOs. We can use this to speed up the VQE

    # Construct the UCCSD circuit. This can be built using the ucc_ansatz function
    circuit = ucc_ansatz(mol, excitations=excitations)

    # Draw the circuit if interested
    # circuit.draw()
    # print()

    # By default, the ucc_ansatz function already sets the circuit parameters to be MP2 guess parameters
    # We can extract the guess parameters for each unique excitation from the circuit itself
    params = []
    for (param,) in circuit.get_parameters():
        # Single excitations will have a zero guess. We will add those in later
        if param and not any(np.isclose(abs(param.real), _param) for _param in params):
            params.append(abs(param.real))
    params += [0.0, 0.0]  # Two unique single excitations
    print(f"MP2 guess parameters: {params}")

    best, params, _extra = optimize(electronic_energy, params, args=(hamiltonian, circuit, excitations))

    print("\nResults using Qibo optimize: (With HF embedding)")
    # Reference energy is the exact ground state
    fci_energy = hamiltonian.eigenvalues()[0]
    print(f"FCI energy: {fci_energy:.8f}")
    print(f" HF energy: {mol.e_hf:.8f}")
    print(f"VQE energy: {best:.8f} (UCCSD ansatz)")
    # print()
    # print("Optimized parameters:", params)


if __name__ == "__main__":
    main()
