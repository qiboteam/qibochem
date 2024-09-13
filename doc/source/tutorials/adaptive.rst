Adaptive methods
================

Despite VQE being cheaper than QPE, circuit depth is still a big problem for today's quantum hardware.

.. code-block:: python

    from qibochem.driver import Molecule
    from qibochem.ansatz import ucc_ansatz

    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    mol.run_pyscf()
    mol.hf_embedding(active=[0, 1, 2, 5])
    circuit = ucc_ansatz(mol)
    print(circuit.summary())

Output:

.. code-block:: output

    Circuit depth = 1874
    Total number of gates = 3300
    Number of qubits = 8
    Most common gates:
    cx: 1312
    h: 1216
    sdg: 304
    s: 304
    rz: 160
    x: 4

As shown in the above code block, the full UCCSD circuit for a simplified LiH/STO-3G system has a circuit depth of 1874 (!), with more than 1000 CNOT gates required!
Hence, there is still a need to further reduce and simplify the circuit ansatzes used for running a VQE simulation.

Other than designing shorter and more efficient circuit ansatzes, one alternative approach is through the use of energy gradients - for instance, through the Parameter Shift Rule on hardware - to filter and reduce the number of fermionic excitations in a circuit ansatz. [#f1]_ [#f2]_
This is known as an adaptive method, in the sense that the quantum gates used to construct the circuit ansatz, as well as its actual structure and arrangement is not fixed, and varies depending on the molecular system under study.

For example, in a H2/STO-3G system mapped with the Jordan-Wigner transformation, there are three possible spin-allowed fermionic excitations:
two single excitations (``[0, 2]``, ``[1, 3]``) and one double excitation (``[0, 1, 2, 3]``).
The full UCCSD circuit for this system has been shown in an earlier :ref:`example <UCC Ansatz>`, and it requires 64 CNOT gates for this simple molecular system.

Let's look at the gradients of each of the individual fermionic excitations:

.. code-block:: python

    from qibo.derivative import parameter_shift

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit, ucc_circuit

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf()
    hamiltonian = mol.hamiltonian()

    excitations = [[0, 1, 2, 3], [0, 2], [1, 3]]
    circuit = hf_circuit(4, 2)
    for excitation in excitations:
        _circuit = circuit.copy()
        _circuit += ucc_circuit(4, excitation)
        n_parameters = len(_circuit.get_parameters())
        gradients = [round(parameter_shift(_circuit, hamiltonian, index), 5) for index in range(n_parameters)]
        print(f"Energy gradients for {excitation}: {gradients}")

Output:

.. code-block:: output

    Energy gradients for [0, 1, 2, 3]: [0.179, -0.179, -0.179, -0.179, 0.179, 0.179, 0.179, -0.179]
    Energy gradients for [0, 2]: [0.0, 0.0]
    Energy gradients for [1, 3]: [0.0, 0.0]

Only the doubles excitation has a non-zero energy gradient, which follows Brillouin's theorem.
Running the VQE with only the doubles excitation then gives:

.. code-block:: python

    import numpy as np

    from qibo.models import VQE

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit, ucc_circuit

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf()
    hamiltonian = mol.hamiltonian()

    circuit = hf_circuit(4, 2)
    circuit += ucc_circuit(4, [0, 1, 2, 3])

    vqe = VQE(circuit, hamiltonian)

    # Optimize starting from a random guess for the variational parameters
    initial_parameters = np.random.uniform(0, 2*np.pi, len(circuit.get_parameters()))
    best, params, extra = vqe.minimize(initial_parameters, method='BFGS', compile=False)

    # Exact result
    print(f"Exact result: {mol.eigenvalues(hamiltonian)[0]:.7f}")
    print(f"  VQE result: {best:.7f}")

Output:

.. code-block:: output

    Exact result: -1.1361895
      VQE result: -1.1361895

We managed to find the exact result by applying only the doubles excitation!

Next, let's look at the potential savings for the simplified LiH/STO-3G system.
To reduce the circuit depth further, we will use the more modern ansatz, the Givens excitation circuit from Arrazola et al., [#f1]_ instead of the UCC ansatz.

As was done in the above example, we will start with a HF circuit, then find the gradients for each circuit ansatz corresponding to a fermionic excitation.
After that, the excitation with the largest absolute value of the gradient will be added to the initial circuit, followed by a VQE simulation.

.. code-block:: python

    from qibo.derivative import parameter_shift
    from qibo.models import VQE

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit, givens_excitation_circuit, generate_excitations, sort_excitations

    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    mol.run_pyscf()
    mol.hf_embedding(active=[0, 1, 2, 5])
    hamiltonian = mol.hamiltonian()

    n_qubits = mol.n_active_orbs
    n_elec = mol.n_active_e

    circuit = hf_circuit(n_qubits, n_elec)

    excitations = sort_excitations(generate_excitations(2, list(range(n_elec)), list(range(n_elec, n_qubits))))
    excitations += sort_excitations(generate_excitations(1, list(range(n_elec)), list(range(n_elec, n_qubits))))

    excitation_gradients = {}
    for excitation in excitations:
        _circuit = circuit.copy()
        _circuit += givens_excitation_circuit(n_qubits, excitation)
        n_parameters = len(_circuit.get_parameters())
        gradient = [round(parameter_shift(_circuit, hamiltonian, index), 5) for index in range(n_parameters)]
        print(f"Energy gradients for {excitation}: {gradient}")
        excitation_gradients[tuple(excitation)] = gradient[0] # Gradient magnitude is equal throughout

    # Find the excitation corresponding to the largest gradient, and add it to the circuit
    max_grad = max(excitation_gradients, key=lambda x: abs(excitation_gradientis.get(x)))
    print(f"\nExcitation with the largest gradient: {max_grad}; Gradient = {excitation_gradients[max_grad]}")
    circuit += givens_excitation_circuit(n_qubits, max_grad)

    # Run VQE with the updated circuit
    vqe = VQE(circuit, hamiltonian)

    circuit_parameters = [param for _tuple in circuit.get_parameters() for param in _tuple]
    best, params, extra = vqe.minimize(circuit_parameters, method='BFGS', compile=False)

    print(f" HF energy: {mol.e_hf:.7f}")
    print(f"VQE result: {best:.7f}")

Output:

.. code-block:: output

    Energy gradients for [0, 1, 4, 5]: [0.02132, -0.02132, 0.02132, -0.02132, -0.02132, 0.02132, -0.02132, 0.02132]
    Energy gradients for [0, 1, 6, 7]: [0.00569, -0.00569, 0.00569, -0.00569, -0.00569, 0.00569, -0.00569, 0.00569]
    Energy gradients for [2, 3, 4, 5]: [0.01136, -0.01136, 0.01136, -0.01136, -0.01136, 0.01136, -0.01136, 0.01136]
    Energy gradients for [2, 3, 6, 7]: [0.12225, -0.12225, 0.12225, -0.12225, -0.12225, 0.12225, -0.12225, 0.12225]
    Energy gradients for [0, 1, 4, 7]: [0.00016, -0.00016, 0.00016, -0.00016, -0.00016, 0.00016, -0.00016, 0.00016]
    Energy gradients for [0, 1, 5, 6]: [-0.00016, 0.00016, -0.00016, 0.00016, 0.00016, -0.00016, 0.00016, -0.00016]
    Energy gradients for [2, 3, 4, 7]: [-0.03254, 0.03254, -0.03254, 0.03254, 0.03254, -0.03254, 0.03254, -0.03254]
    Energy gradients for [2, 3, 5, 6]: [0.03254, -0.03254, 0.03254, -0.03254, -0.03254, 0.03254, -0.03254, 0.03254]
    Energy gradients for [0, 3, 4, 5]: [0.00029, -0.00029, 0.00029, -0.00029, -0.00029, 0.00029, -0.00029, 0.00029]
    Energy gradients for [1, 2, 4, 5]: [-0.00029, 0.00029, -0.00029, 0.00029, 0.00029, -0.00029, 0.00029, -0.00029]
    Energy gradients for [0, 3, 6, 7]: [0.00108, -0.00108, 0.00108, -0.00108, -0.00108, 0.00108, -0.00108, 0.00108]
    Energy gradients for [1, 2, 6, 7]: [-0.00108, 0.00108, -0.00108, 0.00108, 0.00108, -0.00108, 0.00108, -0.00108]
    Energy gradients for [0, 2, 4, 6]: [0.00299, -0.00299, 0.00299, -0.00299, -0.00299, 0.00299, -0.00299, 0.00299]
    Energy gradients for [1, 3, 5, 7]: [0.00299, -0.00299, 0.00299, -0.00299, -0.00299, 0.00299, -0.00299, 0.00299]
    Energy gradients for [0, 3, 4, 7]: [-0.00236, 0.00236, -0.00236, 0.00236, 0.00236, -0.00236, 0.00236, -0.00236]
    Energy gradients for [0, 3, 5, 6]: [-0.00063, 0.00063, -0.00063, 0.00063, 0.00063, -0.00063, 0.00063, -0.00063]
    Energy gradients for [1, 2, 4, 7]: [-0.00063, 0.00063, -0.00063, 0.00063, 0.00063, -0.00063, 0.00063, -0.00063]
    Energy gradients for [1, 2, 5, 6]: [-0.00236, 0.00236, -0.00236, 0.00236, 0.00236, -0.00236, 0.00236, -0.00236]
    Energy gradients for [0, 4]: [0.0, -0.0]
    Energy gradients for [1, 5]: [-0.0, 0.0]
    Energy gradients for [0, 6]: [0.0, -0.0]
    Energy gradients for [1, 7]: [-0.0, 0.0]
    Energy gradients for [2, 4]: [-0.0, 0.0]
    Energy gradients for [3, 5]: [0.0, -0.0]
    Energy gradients for [2, 6]: [-0.0, 0.0]
    Energy gradients for [3, 7]: [0.0, -0.0]

    Excitation with the largest gradient: (2, 3, 6, 7); Gradient = 0.12225
     HF energy: -7.8605387
    VQE result: -7.8732886


After adding the circuit ansatz corresponding to one double excitation and running VQE,
the resultant energy was found to be about 0.01 Hartrees lower compared to the bare HF ansatz.
Clearly, there is still room for further improvement in the obtained energies.

To do so, we further extend the circuit ansatz by adding in more excitations iteratively,
with the excitation with the largest (in magnitute) energy gradients added on at each step.
This can be carried out until the difference between each iteration is small (<0.001 Hartrees), or until there are no more remaining excitations to be added.

.. warning::

      The code block below might take a few minutes to run!

.. code-block:: python

    import numpy as np

    from qibo.models import VQE
    from qibo.derivative import parameter_shift

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit, givens_excitation_circuit, generate_excitations, sort_excitations

    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
    mol.run_pyscf()
    mol.hf_embedding(active=[0, 1, 2, 5])
    hamiltonian = mol.hamiltonian()

    exact_result = mol.eigenvalues(hamiltonian)[0]

    n_qubits = mol.n_active_orbs
    n_elec = mol.n_active_e

    circuit = hf_circuit(n_qubits, n_elec)

    excitations = sort_excitations(generate_excitations(2, list(range(n_elec)), list(range(n_elec, n_qubits))))
    excitations += sort_excitations(generate_excitations(1, list(range(n_elec)), list(range(n_elec, n_qubits))))

    excitation_gradient = {tuple(excitation):0.0 for excitation in excitations}
    count = 0
    current_energy = mol.e_hf
    n_fixed_params = 0
    # Iterating through all excitations; loop breaks if difference in VQE result between excitations is v small
    while excitations:
        print(f"Iteration {count+1}:")
        for excitation in excitations:
            _circuit = circuit.copy()
            _circuit += givens_excitation_circuit(n_qubits, excitation)
            n_parameters = len(_circuit.get_parameters())
            gradient = [round(parameter_shift(_circuit, hamiltonian, index), 5) for index in range(n_parameters)][n_fixed_params:]
            # print(f"Energy gradient for {excitation}: {gradient}")
            excitation_gradient[tuple(excitation)] = gradient[0] # Gradient magnitude is equal throughout

        # Find the excitation corresponding to the largest gradient, and add it to the circuit
        max_grad = max(excitation_gradient, key=lambda x: abs(excitation_gradient.get(x)))
        print(f"Excitation with the largest gradient: {max_grad}; Gradient = {excitation_gradient[max_grad]}")
        circuit += givens_excitation_circuit(n_qubits, max_grad)
        # Remove max_grad from excitations and excitation_data
        excitations.pop(excitations.index(list(max_grad)))
        del excitation_gradient[max_grad]

        # Run VQE with the updated circuit
        vqe = VQE(circuit, hamiltonian)

        circuit_parameters = [param for _tuple in circuit.get_parameters() for param in _tuple]
        best, params, extra = vqe.minimize(circuit_parameters, method='BFGS', compile=False)

        n_fixed_params = len(params)

        print(f"\nExact result: {exact_result:.7f}")
        print(f"  VQE result: {best:.7f}")
        energy_diff = best - current_energy
        print(f"Difference to previous result: {energy_diff:.7f}")

        if abs(energy_diff) < 1e-3:
            print("\nEnergy has converged; exiting while loop")
            break
        print()
        # Update circuit parameters and current energy
        circuit.set_parameters(params)
        current_energy = best
        count += 1


    print("\nFinal circuit:")
    print(circuit.draw())
    print("\nCircuit statistics:")
    print(circuit.summary())

Output:

.. code-block:: output

    Iteration 1:
    Excitation with the largest gradient: (2, 3, 6, 7); Gradient = 0.12225

    Exact result: -7.8770974
      VQE result: -7.8732886
    Difference to previous result: -0.0127499

    Iteration 2:
    Excitation with the largest gradient: (2, 3, 4, 7); Gradient = -0.03485

    Exact result: -7.8770974
      VQE result: -7.8748417
    Difference to previous result: -0.0015531

    Iteration 3:
    Excitation with the largest gradient: (2, 3, 5, 6); Gradient = 0.03364

    Exact result: -7.8770974
      VQE result: -7.8762910
    Difference to previous result: -0.0014493

    Iteration 4:
    Excitation with the largest gradient: (0, 1, 4, 5); Gradient = 0.02124

    Exact result: -7.8770974
      VQE result: -7.8763762
    Difference to previous result: -0.0000853

    Energy has converged; exiting while loop

    Final circuit:
    q0:     ─X──────────────────────────────────────────────────────────────────── ...
    q1:     ─X──────────────────────────────────────────────────────────────────── ...
    q2:     ─X───o─H─o───RY─o─────RY───X─RY─────o─RY─o─X─H─o─────o─H─o───RY─o───── ...
    q3:     ─X───|───X───RY─|───X─RY─X─|─RY─X───|─RY─X─|───|─────|───X───RY─|───X─ ...
    q4:     ─────|──────────|───|────|─|────|───|──────|───|───o─X─────o────|───|─ ...
    q5:     ─────|──────────|───|────|─|────|───|──────|───|───|───────|────|───|─ ...
    q6:     ───o─X─────o────|───|────o─o────|───|──────o───X─o─|───────|────|───|─ ...
    q7:     ───X───H───X────X─H─o───────────o─H─X────────H───X─X───H───X────X─H─o─ ...

    q0: ... ────────────────────────────────────────────────────────────────────── ...
    q1: ... ────────────────────────────────────────────────────────────────────── ...
    q2: ... RY───X─RY─────o─RY─o─X─H─o─────o─H─o───RY─o─────RY───X─RY─────o─RY─o─X ...
    q3: ... RY─X─|─RY─X───|─RY─X─|───|─────|───X───RY─|───X─RY─X─|─RY─X───|─RY─X─| ...
    q4: ... ───o─o────|───|──────o───X─o───|──────────|───|────|─|────|───|──────| ...
    q5: ... ──────────|───|────────────|─o─X─────o────|───|────o─o────|───|──────o ...
    q6: ... ──────────|───|────────────|─X───H───X────X─H─o───────────o─H─X─────── ...
    q7: ... ──────────o─H─X────────H───X────────────────────────────────────────── ...

    q0: ... ─────────o─H─o───RY─o─────RY───X─RY─────o─RY─o─X─H─o───
    q1: ... ─────────|───X───RY─|───X─RY─X─|─RY─X───|─RY─X─|───|───
    q2: ... ─H─o─────|──────────|───|────|─|────|───|──────|───|───
    q3: ... ───|─────|──────────|───|────|─|────|───|──────|───|───
    q4: ... ───|───o─X─────o────|───|────o─o────|───|──────o───X─o─
    q5: ... ───X─o─X───H───X────X─H─o───────────o─H─X────────H───X─
    q6: ... ─H───X─────────────────────────────────────────────────
    q7: ... ───────────────────────────────────────────────────────

    Circuit statistics:
    Circuit depth = 78
    Total number of gates = 116
    Number of qubits = 8
    Most common gates:
    cx: 56
    ry: 32
    h: 24
    x: 4

Recall that the full UCCSD circuit for our system had a circuit depth of 1874, with more than 1000 CNOT gates required.
In contrast, the use of a simpler circuit ansatz in conjunction with an adaptive approach allowed us to find a VQE energy that is within chemical accuracy,
while using only 56 CNOT gates and with a final gate depth of only 78.
This is a >20-fold reduction in the gate depth and number of CNOT gates used!


  .. rubric:: References

  .. [#f1] Arrazola, J. M. et al., 'Universal quantum circuits for quantum chemistry', Quantum, 6, (2022), 742.

  .. [#f2] Schuld M. et al., 'Evaluating analytic gradients on quantum hardware', Phys. Rev. A, 99, (2019), 032331.
