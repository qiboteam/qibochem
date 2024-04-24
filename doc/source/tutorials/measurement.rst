Expectation from samples
========================

The previous examples were all carried out using state vector simulations of the quantum circuit.
However, in actual quantum hardware, the expectation value of the molecular Hamiltonian for a parameterized quantum circuit has to be estimated using repeated executions of the circuit, or shots in short.

.. code-block:: python

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit
    from qibochem.measurement import expectation, expectation_from_samples

    # Build the H2 molecule and get the molecular Hamiltonian
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    hamiltonian = h2.hamiltonian()
    print(f"Number of terms in the Hamiltonian: {len(hamiltonian.terms)}")

    # Construct a basic Hartree-Fock circuit
    circuit = hf_circuit(h2.nso, h2.nelec)

    # Expectation value using a state vector simulation:
    exact_result = expectation(circuit, hamiltonian)
    # Expectation value using (simulated) shots
    shots_result = expectation_from_samples(circuit, hamiltonian, n_shots=1000)
    print(f"\nExact result: {exact_result:.8f}")
    # There will be a small difference between the exact result and the results with shots
    print(f"Shots result: {shots_result:.8f}")


.. code-block:: output

    Number of terms in the Hamiltonian: 14

    Exact result: -1.11734903
    Shots result: -1.11260552


In the case of the H\ :sub:`2`/STO-3G (4 qubit) example above, there are 14 terms that comprise the molecular Hamiltonian.
In practice, the expectation value for each of the individual Pauli terms have to be obtained using circuit measurements, before summing them up to obtain the overall expectation value of the molecular Hamiltonian.

This process of obtaining the electronic energy (Hamiltonian expectation value) is still reasonable for a small system.
However, the number of Pauli terms in a molecular Hamiltonian scales on the order of :math:`O(N^4)`, where N is the number of qubits.

.. code-block:: python

    from qibochem.driver import Molecule

    # Build the N2 molecule and get the molecular Hamiltonian
    n2 = Molecule([("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.1))])
    n2.run_pyscf()
    hamiltonian = n2.hamiltonian()
    print(f"Number of terms in the Hamiltonian: {len(hamiltonian.terms)}")


.. code-block:: output

    Number of terms in the Hamiltonian: 2950


Even for the relatively small N\ :sub:`2` molecule with the minimal STO-3G basis set, there are already 2950 (!) terms to measure.
Going further, if the electronic energy is evaluated as part of the process of running a VQE, it has to be repeated for each step of the VQE.
Clearly, the measurement cost of running VQE has the potential to become astronomically large, and is a significant practical challenge today.


Reducing the measurement cost
-----------------------------

So far, we have assumed that the Hamiltonian expectation values have to be obtained using an independent set of circuit measurements for each term in the molecular Hamiltonian.
However, we know from quantum mechanics that if two observables (the indvidual Pauli terms in the Hamiltonian) commute, they can be measured simultaneously.
More precisely, if two observables commute, they have a common eigenbasis, i.e.

.. math::

    [A, B] = 0 \implies \exists \underset{~}{x} \text{ such that } A \underset{~}{x} = a \underset{~}{x}  \text{ and } B \underset{~}{x} = b \underset{~}{x}

In other words, a single set of measurements, carried out in the common eigenbasis, can be used to obtain the expectation values of two (or more) commuting observables simultaneously.
What remains is then how to apply the above fact towards the reduction of the measurement cost in practice.


Grouping Hamiltonian terms
--------------------------

First, there is the question of how to sort the Hamiltonian terms into separate groups of mutually commuting terms; i.e. each term in a group commutes with every other term in the same group.
Less groups would mean that a smaller number of measurements are required, which is our eventual goal:

.. Picture of graphs with commuting terms


In the above example, blah blah complete graphs and blah blah, duno what can commute with dunno what and dunno what, but it would be better if so and so was grouped with so and so.
This problem of finding the smallest possible number of groups is equivalent to the minimum clique cover problem, i.e. finding the smallest number of cliques (groups) of complete graphs.



PennyLane: "Unfortunately, that’s where our good fortune ends—the minimum clique cover problem is known to be NP-hard, meaning there is no known (classical) solution to finding the optimum/minimum clique cover in polynomial time.
Thankfully, there is a silver lining: we know of polynomial-time algorithms for finding approximate solutions to the minimum clique cover problem"
, and these algorithms are available in the NetworkX library:

.. Example for H for some system

Qubit-wise commuting terms
--------------------------

After obtaining groups of mutually commuting observables, it remains to find the shared eigenbasis for all terms in the group, and to prepare a set of measurements carried out in this basis.
Unfortunately, this is not trivial: Need to diagonalize matrix here there, combine each eigenvector, blah blah.

However, if the stricter condition of qubit-wise commutativty is enforced, it becomes simple to obtain the shared eigenbasis.



Putting everything together
---------------------------

ZC note: Can put the text from the current example here. Show how much Hamiltonian cost reduced for electronic energy evaluation, then extend to each step in VQE.

.. Code with individual functions

For convenience, the above has been combined into the ``expectation_from_samples`` function (add link)

.. Code calling expectation_from_sample directly


Final notes
-----------

(New): Lastly, it may be possible that using a single set of measurements may be undesirable due to errors and uncertainty in the measurement results being propagated across a number of terms.
If a single set of measurements are used for an individual Pauli term, any issues with this set of measurements would not extend to the expectation value of the other Hamiltonian terms.
There are some suggestions towards mitigating this issue. (ref)


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
