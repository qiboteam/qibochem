"""
Sample script for using the physics-constrained hardware-efficient ansatz. Helper functions help set the circuit
parameters and calculate the energy expectation value

Reference: https://pubs.acs.org/doi/10.1021/acs.jctc.3c00966
"""

import numpy as np
from qibo.optimizers import optimize

from qibochem.ansatz import pche_circuit
from qibochem.driver import Molecule


def single_layer_parameters(parameters: np.ndarray, nqubits: int) -> list[float]:
    """Extend parameters to fit the circuit parameters for a single layer of the PC-HEA"""
    # Get U1 (RY and RX) parameters directly
    u1_parameters = parameters[: 2 * nqubits].tolist()
    # U2 parameters: Ry-f-Ry: Ry(phi/2), fSim(theta, phi), Ry(-phi/2). Order in parameters set as phi then theta
    u2_parameters = [
        [
            0.5 * parameters[2 * nqubits + 2 * i],
            parameters[2 * nqubits + 2 * i + 1],
            parameters[2 * nqubits + 2 * i],
            -0.5 * parameters[2 * nqubits + 2 * i],
        ]
        for i in range(nqubits - 1)
    ]
    # U2.dagger() parameters (Ry-f-Ry) by reversing and taking negative of u2_parameters
    u2_dag_parameters = [[-param for param in gate_param] for gate_param in reversed(u2_parameters)]
    # Flatten out both lists
    u2_parameters = [param for gate in u2_parameters for param in gate]
    u2_dag_parameters = [param for gate in u2_dag_parameters for param in gate]
    # RZ parameters
    rz_parameters = parameters[2 * nqubits + 2 * (nqubits - 1) :].tolist()
    # U1.dagger() (RY and RX) parameters
    u1_dag_parameters = [-param for param in u1_parameters[nqubits:]] + [-param for param in u1_parameters[:nqubits]]
    circuit_parameters = u1_parameters + u2_parameters + rz_parameters + u2_dag_parameters + u1_dag_parameters
    return circuit_parameters


def energy(parameters, circuit, hamiltonian, nlayers, nqubits):
    """Expectation value for hamiltonian given some quantum circuit"""
    # No. of parameters for a single layer
    n_parameters = 2 * nqubits + 2 * (nqubits - 1) + nqubits  # U1, U2, and RZ respectively

    circuit_parameters = []
    for i in range(nlayers):
        circuit_parameters += single_layer_parameters(parameters[i * (n_parameters) : (i + 1) * n_parameters], nqubits)
    circuit.set_parameters(circuit_parameters)
    return hamiltonian.expectation(circuit)


def main():
    """Main function"""
    # Build molecule
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.75))])
    mol.run_pyscf()
    hamiltonian = mol.hamiltonian()

    # Define circuit ansatz
    nqubits = hamiltonian.nqubits
    nlayers = 2
    circuit = pche_circuit(nqubits, nlayers)

    print("Physics-Constrained HEA:")
    circuit.draw()
    print()

    fci_energy = hamiltonian.eigenvalues()[0]

    params = np.random.rand(len(circuit.get_parameters()))
    best, params, _extra = optimize(energy, params, args=(circuit, hamiltonian, nlayers, nqubits))

    print("\nResults (With constrained parameters):")
    print(f"FCI energy: {fci_energy:.8f}")
    print(f" HF energy: {mol.e_hf:.8f} (Hartree-Fock)")
    print(f"VQE energy: {best:.8f} (PC-HEA)")


if __name__ == "__main__":
    main()
