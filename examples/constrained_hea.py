"""
Draft code for physics constrained hardware-efficient ansatz. Ref: https://pubs.acs.org/doi/10.1021/acs.jctc.3c00966
"""

import numpy as np
from qibo import Circuit, gates
from qibo.optimizers import optimize

from qibochem.driver import Molecule
from qibochem.measurement import expectation


def pchea_circuit(n_qubits, n_layers):
    """Construct physics constraited HEA"""
    circuit = Circuit(n_qubits)
    assert n_layers, "n_layers must be >0!"

    for _i in range(n_layers):
        # Add U1 to each qunit
        circuit.add(gates.RY(_i, 0.0) for _i in range(n_qubits))
        circuit.add(gates.RX(_i, 0.0) for _i in range(n_qubits))
        # pchea_ansatz.apply(RY(u1_thetas[_i]), qubit_register[_i])
        # pchea_ansatz.apply(RX(u1_thetas[n_qubits+_i]), qubit_register[_i])

        # Add U2 to entangle each qubit
        for _i in range(n_qubits - 1):
            circuit.add(gates.RY(_i, 0.0))
            circuit.add(gates.fSim(_i, _i + 1, 0.0, 0.0))
            circuit.add(gates.RY(_i, 0.0))
        # for _i in range(n_qubits - 1):
        #     pchea_ansatz.apply(RY(0.5*u2_thetas[_i]), qubit_register[_i])
        #     pchea_ansatz.apply(fSim(u2_thetas[_i+n_qubits], u2_thetas[_i]), qubit_register[_i], qubit_register[_i+1])
        #     pchea_ansatz.apply(RY(-0.5*u2_thetas[_i]), qubit_register[_i])

        # Add RZ gates
        circuit.add(gates.RZ(_i, 0.0) for _i in range(n_qubits))
        # for _i in range(n_qubits):
        #     pchea_ansatz.apply(RZ(rz_thetas[_i]), qubit_register[_i])

        # Add U2 to entangle each qubit
        for _i in range(n_qubits - 1, 0, -1):
            circuit.add(gates.RY(_i - 1, 0.0).dagger())
            circuit.add(gates.fSim(_i, _i - 1, 0.0, 0.0).dagger())
            circuit.add(gates.RY(_i - 1, 0.0).dagger())
        # for _i in range(n_qubits - 1, 0, -1):
        #     pchea_ansatz.apply(RY(0.5*u2_thetas[_i]).dag(), qubit_register[_i-1])
        #     pchea_ansatz.apply(fSim(u2_thetas[_i+n_qubits], u2_thetas[_i]).dag(), qubit_register[_i], qubit_register[_i-1])
        #     pchea_ansatz.apply(RY(-0.5*u2_thetas[_i]).dag(), qubit_register[_i-1])

        # Add U1 to each qunit
        circuit.add(gates.RX(_i, 0.0).dagger() for _i in range(n_qubits))
        circuit.add(gates.RY(_i, 0.0).dagger() for _i in range(n_qubits))
        # for _i in range(n_qubits):
        #     pchea_ansatz.apply(RY(u1_thetas[2*n_qubits+_i]), qubit_register[_i])
        #     pchea_ansatz.apply(RX(u1_thetas[3*n_qubits+_i]), qubit_register[_i])

    return circuit


def single_layer_parameters(parameters, n_qubits):
    """Extend parameters to fit the circuit parameters for a single layer of the QC-HEA"""
    # parameters argument should be a np.array
    # Get U1 (RY and RX) parameters directly
    u1_parameters = parameters[: 2 * n_qubits].tolist()
    # U2 parameters: Ry-f-Ry: Ry(phi/2), fSim(theta, phi), Ry(-phi/2). Let order in parameters be phi then theta
    _u2_parameters = [
        [
            0.5 * parameters[2 * n_qubits + 2 * _i],
            parameters[2 * n_qubits + 2 * _i + 1],
            parameters[2 * n_qubits + 2 * _i],
            -0.5 * parameters[2 * n_qubits + 2 * _i],
        ]
        for _i in range(n_qubits - 1)
    ]
    u2_parameters = [_x for _gate in _u2_parameters for _x in _gate]
    # RZ parameters
    rz_parameters = parameters[2 * n_qubits + 2 * (n_qubits - 1) :].tolist()
    # U2.dagger() parameters (Ry-f-Ry) by reversing and taking negative
    _u2_dag_parameters = [[-param for param in gate_param] for gate_param in reversed(_u2_parameters)]
    u2_dag_parameters = [_x for _gate in _u2_dag_parameters for _x in _gate]
    # U1.dagger() (RY and RX) parameters
    u1_dag_parameters = [-param for param in u1_parameters[n_qubits:]] + [-param for param in u1_parameters[:n_qubits]]

    circuit_parameters = u1_parameters + u2_parameters + rz_parameters + u2_dag_parameters + u1_dag_parameters
    return circuit_parameters


def energy(parameters, circuit, hamiltonian, n_layers, n_qubits):
    """Expectation value for hamiltonian given some quantum circuit"""
    # No. of parameters for a single layer
    n_parameters = 2 * n_qubits + 2 * (n_qubits - 1) + n_qubits  # u1  # u2  # RZ
    # print(n_parameters)

    circuit_parameters = []
    for _i in range(n_layers):
        circuit_parameters += single_layer_parameters(
            parameters[_i * (n_parameters) : (_i + 1) * n_parameters], n_qubits
        )
    circuit.set_parameters(circuit_parameters)
    return expectation(circuit, hamiltonian)


# def unconstrained_energy(circuit_parameters, circuit, hamiltonian):
#     circuit.set_parameters(circuit_parameters)
#     return expectation(circuit, hamiltonian)


def main():
    """Main function"""
    # Build molecule
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.75))])
    mol.run_pyscf()
    hamiltonian = mol.hamiltonian()

    # Define circuit ansatz
    n_qubits = hamiltonian.nqubits
    n_layers = 2
    circuit = pchea_circuit(n_qubits, n_layers)
    circuit.draw()
    print()
    # for _gate in circuit.queue:
    #     print(_gate)

    # q0: ─RY─RX─RY─f─RY────────RZ──────────────RY─f─RY─RX─RY─
    # q1: ─RY─RX────f─RY─f─RY───RZ─────────RY─f─RY─f─RX─RY────
    # q2: ─RY─RX─────────f─RY─f─RY─RZ─RY─f─RY─f──────RX─RY────
    # q3: ─RY─RX──────────────f─RZ───────f───────────RX─RY────

    # Circuit parameters:
    # u1: 2*n_qubits, latter half of U1(theta, phi) are all dagger (negative)
    # u2: Ry-f-Ry: Ry(phi/2), fSim(theta, phi), Ry(-phi/2)
    # RZ: No restriction

    # Test function
    # params = np.random.rand(n_parameters)
    # print(energy(params))

    # fci_energy = hamiltonian.eigenvalues()[0]
    fci_energy = -1.13618945

    params = np.random.rand(len(circuit.get_parameters()))
    # params = np.zeros(n_layers*n_parameters)
    print(f"# parameters: {len(circuit.get_parameters())}")

    # energy(parameters, circuit, hamiltonian, n_layers, n_qubits)
    best, params, _extra = optimize(energy, params, args=(circuit, hamiltonian, n_layers, n_qubits))

    print("\nResults (With constrained parameters):")
    print(f"FCI energy: {fci_energy:.8f}")
    print(f" HF energy: {mol.e_hf:.8f} (Classical)")
    print(f"VQE energy: {best:.8f} (QC-HEA)")

    # print()
    # for gate, gate_params in zip(circuit.queue, circuit.get_parameters()):
    #     print(f"{gate.name}{gate.target_qubits}", gate_params)

    n_circuit_params = len(
        [
            param
            for gate_params in circuit.get_parameters()
            for param in (gate_params if hasattr(gate_params, "__iter__") else [param])
        ]
    )
    print(f"# Circuit parameters: {n_circuit_params}")

    # quit()
    #
    # params = np.random.rand(n_circuit_params)
    # # print(len(circuit.get_parameters()))
    # # params = np.zeros(len(circuit.get_parameters()))
    #
    # best, params, extra = optimize(
    #     unconstrained_energy,
    #     params,
    #     args=(circuit, hamiltonian)
    # )
    #
    #
    # print("\nResults (With constrained parameters):")
    # print(f"FCI energy: {fci_energy:.8f}")
    # print(f" HF energy: {mol.e_hf:.8f} (Classical)")
    # print(f"VQE energy: {best:.8f} (QC-HEA)")


if __name__ == "__main__":
    main()
