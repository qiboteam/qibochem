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

Other than designing shorter and more efficient circuit ansatzes, one alternative approach is through the use of energy gradients - for instance, through the Parameter Shift Rule on hardware - to filter and reduce the number of fermionic excitations in a circuit ansatz.  (REFS!!!)
This is known as an adaptive method, in the sense that the quantum gates used to construct the circuit ansatz, as well as its actual structure and arrangement is not fixed, and varies depending on the molecular system under study.

For example, in a H2/STO-3G system mapped with the Jordan-Wigner transformation, there are three possible spin-allowed fermionic excitations:
two single excitations (``[0, 2]``, ``[1, 3]``) and one double excitation (``[0, 1, 2, 3]``).
The full UCCSD circuit for this system has been shown in an earlier example (ref), and it requires 64 CNOT gates for this simple molecular system.

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
To reduce the circuit depth further, we will use the more modern ansatz, the Givens excitation circuit from Arrazola et al., instead of the UCC ansatz.

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


.. comment
  OLD STUFF
  ---------

  A quantum circuit comprising parameterized gates (`e.g.` :math:`RX(\theta)`, :math:`RY(\theta)` and :math:`RZ(\theta)`),
  represents a unitary transformation :math:`U(\theta)` that transforms some initial quantum state into a parametrized ansatz state :math:`|\psi(\theta)\rangle`.

  Examples of some ansatzes available in Qibochem are described in the subsections below.

  Hardware Efficient Ansatz
  -------------------------

  Qibochem provides a hardware efficient ansatz that simply consists of a layer of single-qubit rotation gates followed by a layer of two-qubit gates that entangle the qubits.
  For the H\ :sub:`2` case discussed in previous sections, a possible hardware efficient circuit ansatz can be constructed as such:

  .. image:: qibochem_doc_ansatz_hardware-efficient.svg

  .. code-block:: python

      from qibochem.ansatz import he_circuit

      nqubits = 4
      nlayers = 1

      circuit = he_circuit(nqubits, nlayers)
      print(circuit.draw())

  .. code-block:: output

      q0: ─RY─RZ─o─────Z─
      q1: ─RY─RZ─Z─o───|─
      q2: ─RY─RZ───Z─o─|─
      q3: ─RY─RZ─────Z─o─

  The energy of the state generated from the hardware efficient ansatz for the fermionic two-body Hamiltonian can then be estimated, using state vectors or samples.

  The following example demonstrates how the energy of the H2 molecule is affected with respect to the rotational parameters:

  .. code-block:: python

      import numpy as np

      from qibochem.driver import Molecule
      from qibochem.measurement.expectation import expectation
      from qibochem.ansatz import he_circuit

      mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74804))])
      mol.run_pyscf()
      hamiltonian = mol.hamiltonian()

      # Define and build the HEA
      nlayers = 1
      nqubits = mol.nso
      ntheta = 2 * nqubits * nlayers
      circuit = he_circuit(nqubits, nlayers)

      print("Energy expectation values for thetas: ")
      print("-----------------------------")
      print("| theta | Electronic energy |")
      print("|---------------------------|")
      thetas = [-0.2, 0.0, 0.2]
      for theta in thetas:
          params = np.full(ntheta, theta)
          circuit.set_parameters(params)
          electronic_energy = expectation(circuit, hamiltonian)
          print(f"| {theta:5.1f} | {electronic_energy:^18.12f}|")
      print("-----------------------------")


  .. code-block:: output

      converged SCF energy = -1.11628373627429

      Energy expectation values for thetas:
      -----------------------------
      | theta | Electronic energy |
      |---------------------------|
      |  -0.2 |   0.673325849299  |
      |   0.0 |   0.707418334474  |
      |   0.2 |   0.673325849299  |
      -----------------------------


  .. _UCC Ansatz:

  Unitary Coupled Cluster Ansatz
  ------------------------------

  The Unitary Coupled Cluster (UCC) ansatz [#f1]_ [#f2]_ [#f3]_ is a variant of the popular gold standard Coupled Cluster ansatz [#f4]_ of quantum chemistry.
  The UCC wave function is a parameterized unitary transformation of a reference wave function :math:`\psi_{\mathrm{ref}}`, of which a common choice is the Hartree-Fock wave function.

  .. math::

      \begin{align*}
      |\psi_{\mathrm{UCC}}\rangle &= U(\theta)|\psi_{\mathrm{ref}}\rangle \\
                                  &= e^{\hat{T}(\theta) - \hat{T}^\dagger(\theta)}|\psi_{\mathrm{ref}}\rangle
      \end{align*}


  Similar to the process for the molecular Hamiltonian, the fermionic excitation operators :math:`\hat{T}` and :math:`\hat{T}^\dagger` are mapped using e.g. Jordan-Wigner mapping into Pauli operators.
  This is typically followed by a Suzuki-Trotter decomposition of the exponentials of these Pauli operators, which allows the UCC ansatz to be implemented on quantum computers. [#f5]_

  An example of how to build a UCC doubles circuit ansatz for the :math:`H_2` molecule is given as:

  .. code-block:: python

      from qibochem.driver.molecule import Molecule
      from qibochem.ansatz import hf_circuit, ucc_circuit

      mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74804))])
      mol.run_pyscf()
      hamiltonian = mol.hamiltonian()

      # Set parameters for the rest of the experiment
      n_qubits = mol.nso
      n_electrons = mol.nelec

      # Build UCCD circuit
      circuit = hf_circuit(n_qubits, n_electrons) # Start with HF circuit
      circuit += ucc_circuit(n_qubits, [0, 1, 2, 3]) # Then add the double excitation circuit ansatz

      print(circuit.draw())

  .. code-block:: output

      q0:     ─X──H─────X─RZ─X─────H──RX─────X─RZ─X─────RX─RX─────X─RZ─X─────RX─H─── ...
      q1:     ─X──H───X─o────o─X───H──RX───X─o────o─X───RX─H────X─o────o─X───H──RX── ...
      q2:     ─RX───X─o────────o─X─RX─RX─X─o────────o─X─RX─H──X─o────────o─X─H──H──X ...
      q3:     ─H────o────────────o─H──H──o────────────o─H──H──o────────────o─H──H──o ...

      q0: ... ───X─RZ─X─────H──RX─────X─RZ─X─────RX─H──────X─RZ─X─────H──H──────X─RZ ...
      q1: ... ─X─o────o─X───RX─H────X─o────o─X───H──RX───X─o────o─X───RX─H────X─o─── ...
      q2: ... ─o────────o─X─H──RX─X─o────────o─X─RX─RX─X─o────────o─X─RX─H──X─o───── ...
      q3: ... ────────────o─H──RX─o────────────o─RX─RX─o────────────o─RX─RX─o─────── ...

      q0: ... ─X─────H──RX─────X─RZ─X─────RX─
      q1: ... ─o─X───H──RX───X─o────o─X───RX─
      q2: ... ───o─X─H──H──X─o────────o─X─H──
      q3: ... ─────o─RX─RX─o────────────o─RX─


  ..
     _Basis rotation ansatz

  Basis rotation ansatz
  ---------------------

  The starting points for contemporary quantum chemistry methods are often those based on the mean field approximation within a (finite) molecular orbital basis, i.e. the Hartree-Fock method. The electronic energy is calculated as the mean value of the electronic Hamiltonian :math:`\hat{H}_{\mathrm{elec}}` acting on a normalized single Slater determinant function :math:`\psi` [#f6]_

  .. math::

      \begin{align*}
      E[\psi] &= \langle \psi | \hat{H}_{\mathrm{elec}} |\psi \rangle \\
              &= \sum_i^{N_f} \langle \phi_i |\hat{h}|\phi_i \rangle + \frac{1}{2} \sum_{i,j}^{N_f}
              \langle \phi_i\phi_j||\phi_i\phi_j \rangle
      \end{align*}

  The orthonormal molecular orbitals :math:`\phi` are optimized by a direct minimization of the energy functional with respect to parameters :math:`\kappa` that parameterize the unitary rotations of the orbital basis. Qibochem's implementation uses the QR decomposition of the unitary matrix as employed by Clements et al., [#f7]_ which results in a rectangular gate layout of `Givens rotation gates <https://qibo.science/qibo/stable/api-reference/qibo.html#givens-gate>`_ that yield linear CNOT gate depth when decomposed.


  .. code-block:: python

      import numpy as np
      from qibochem.driver.molecule import Molecule
      from qibochem.ansatz import basis_rotation, ucc
      from qibo import Circuit, gates, models

      def basis_rotation_circuit(mol, parameters=0.0):

          nqubits = mol.nso
          occ = range(0, mol.nelec)
          vir = range(mol.nelec, mol.nso)

          U, kappa = basis_rotation.unitary(occ, vir, parameters=parameters)
          gate_angles, final_U = basis_rotation.givens_qr_decompose(U)
          gate_layout = basis_rotation.basis_rotation_layout(nqubits)
          gate_list, ordered_angles = basis_rotation.basis_rotation_gates(gate_layout, gate_angles, kappa)

          circuit = Circuit(nqubits)
          for _i in range(mol.nelec):
              circuit.add(gates.X(_i))
          circuit.add(gate_list)

          return circuit, gate_angles

      h3p = Molecule([('H', (0.0000,  0.0000, 0.0000)),
                      ('H', (0.0000,  0.0000, 0.8000)),
                      ('H', (0.0000,  0.0000, 1.6000))],
                      charge=1, multiplicity=1)
      h3p.run_pyscf(max_scf_cycles=1)

      e_init = h3p.e_hf
      h3p_sym_ham = h3p.hamiltonian("sym", h3p.oei, h3p.tei, 0.0, "jw")

      hf_circuit, qubit_parameters = basis_rotation_circuit(h3p, parameters=0.1)

      print(hf_circuit.draw())

      vqe = models.VQE(hf_circuit, h3p_sym_ham)
      res = vqe.minimize(qubit_parameters)

      print('energy of initial guess: ', e_init)
      print('energy after vqe       : ', res[0])

  .. code-block:: output

      q0: ─X─G─────────G─────────G─────────
      q1: ─X─G─────G───G─────G───G─────G───
      q2: ─────G───G─────G───G─────G───G───
      q3: ─────G─────G───G─────G───G─────G─
      q4: ───────G───G─────G───G─────G───G─
      q5: ───────G─────────G─────────G─────
      basis rotation: using uniform value of 0.1 for each parameter value
      energy of initial guess:  -1.1977713400022736
      energy after vqe       :  -1.2024564133305427






  .. rubric:: References

  .. [#f1] Kutzelnigg, W. (1977). 'Pair Correlation Theories', in Schaefer, H.F. (eds) Methods of Electronic Structure Theory. Modern Theoretical Chemistry, vol 3. Springer, Boston, MA.

  .. [#f2] Whitfield, J. D. et al., 'Simulation of Electronic Structure Hamiltonians using Quantum Computers', Mol. Phys. 109 (2011) 735.

  .. [#f3] Anand. A. et al., 'A Quantum Computing view on Unitary Coupled Cluster Theory', Chem. Soc. Rev. 51 (2022) 1659.

  .. [#f4] Crawford, T. D. et al., 'An Introduction to Coupled Cluster Theory for Computational Chemists', in Reviews in Computational Chemistry 14 (2007) 33.

  .. [#f5] Barkoutsos, P. K. et al., 'Quantum algorithms for electronic structure calculations: Particle-hole Hamiltonian and optimized wave-function expansions', Phys. Rev. A 98 (2018) 022322.

  .. [#f6] Piela, L. (2007). 'Ideas of Quantum Chemistry'. Elsevier B. V., the Netherlands.

  .. [#f7] Clements, W. R. et al., 'Optimal Design for Universal Multiport Interferometers', Optica 3 (2016) 1460.
