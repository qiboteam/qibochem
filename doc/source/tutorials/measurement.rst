Expectation from samples
------------------------

The previous examples were all carried out using state vector simulations of the quantum circuit.
However, in actual quantum hardware, the expectation value of the molecular Hamiltonian for a parameterized quantum circuit have to be estimated by using repeated executions of the circuit, or shots in short.

.. code-block
   H2/STO-3G, JW Hamiltonian

For example, in the H2/STO-3G system, there are 14 terms that comprise the molecular Hamiltonian,
which means that the expectation value for each of the individual Pauli terms have to be obtained using circuit measurements, before summing them up to obtain the overall expectation value of the molecular Hamiltonian.

This process of obtaining the electronic energy (Hamiltonian expectation value) is still reasonable for a small system.
Unfortunately, the number of terms in a molecular Hamiltonian scales on the order of O(N^4), where N is the number of qubits.

.. code-block
    N2/STO-3G, JW Hamiltonian

Even for a relatively small molecule with the minimal STO-3G basis set, there are already (?!?) terms to measure.
Going further, if the electronic energy is obtained towards the goal of running a VQE, it has to be repeated for each step of the VQE.
Clearly, the measurement cost of running VQE has the potential to become astronomically large, and is a significant practical challenge today.


Reducing the measurement cost
----------------------------

In the examples above, the Hamiltonian expectation values were obtained using a separate set of circuit measurements for each individual term in the molecular Hamiltonian.

However, we know from quantum mechanics that if two observables (the indvidual Pauli terms in the Hamiltonian) commute, they can be measured simultaneously.

.. Some math?


The question is how to use this in practice?




OLD TEXT, TO BE EDITED
----------------------


Qibochem provides this functionality using the :code:`AbstractHamiltonian.expectation_from_samples` method implemented in Qibo.

The example below is taken from the Bravyi-Kitaev transformed Hamiltonian for molecular H\ :sub:`2` in minimal basis of Hartree-Fock orbitals, at 0.70 Angstroms separation between H nuclei,
as was done in [#f1]_:


Hamiltonian expectation value
-----------------------------

.. code-block:: python

    from qibo import models, gates
    from qibo.symbols import X, Y, Z
    from qibo.hamiltonians import SymbolicHamiltonian
    import numpy as np
    from qibochem.measurement.expectation import expectation
    from scipy.optimize import minimize

    # Bravyi-Kitaev tranformed Hamiltonian for H2 at 0.7 Angstroms
    bk_ham_form = -0.4584 + 0.3593*Z(0) - 0.4826*Z(1) + 0.5818*Z(0)*Z(1) + 0.0896*X(0)*X(1) + 0.0896*Y(0)*Y(1)
    bk_ham = SymbolicHamiltonian(bk_ham_form)
    nuc_repulsion = 0.7559674441714287

    circuit = models.Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.RX(0, -np.pi/2, trainable=False))
    circuit.add(gates.RY(1, np.pi/2, trainable=False))
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.RZ(0, theta=0.0))
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.RX(0, np.pi/2, trainable=False))
    circuit.add(gates.RY(1, -np.pi/2, trainable=False))

    print(circuit.draw())

    def energy_expectation_samples(parameters, circuit, hamiltonian, nshots=1024):
        return expectation(circuit, hamiltonian, from_samples=True, n_shots=nshots)

    parameters = [0.5]
    nshots = 8192
    vqe_uccsd = minimize(energy_expectation_samples, parameters, args=(circuit, bk_ham, nshots), method='Powell')
    print(vqe_uccsd)
    print('VQE UCCSD loss:   ', vqe_uccsd.fun)
    print('nuclear repulsion:', nuc_repulsion)
    print('VQE UCCSD energy: ', vqe_uccsd.fun + nuc_repulsion)


.. code-block:: output

    q0: ─X──RX─X─RZ─X─RX─
    q1: ─RY────o────o─RY─
    message: Optimization terminated successfully.
    success: True
    status: 0
        fun: -1.8841124999999999
        x: [ 2.188e+00]
        nit: 2
    direc: [[ 1.000e+00]]
        nfev: 23
    VQE UCCSD loss:    -1.8841124999999999
    nuclear repulsion: 0.7559674441714287
    VQE UCCSD energy:  -1.128145055828571


.. rubric:: References

.. [#f1] P. J. J. O'Malley et al. 'Scalable Quantum Simulation of Molecular Energies' Phys. Rev. X (2016) 6, 031007.
