Ansatz
======

A quantum circuit comprising parameterized gates (`e.g.` :math:`RX(\theta)`, :math:`RY(\theta)` and :math:`RZ(\theta)`),
represents a unitary transformation :math:`U(\theta)` that transforms some initial quantum state into a parametrized ansatz state :math:`|\psi(\theta)\rangle`.

Examples of some ansatzes available in Qibochem are described in the subsections below.

Hardware Efficient Ansatz
-------------------------

Qibochem provides a hardware efficient ansatz that simply consists of a layer of single-qubit rotation gates followed by a layer of two-qubit gates that entangle the qubits.
For the H\ :sub:`2` case discussed in previous sections, a possible hardware efficient circuit ansatz can be constructed as such:

.. image:: qibochem_doc_ansatz_hardware-efficient.svg

.. code-block:: python

    from qibo import Circuit

    from qibochem.ansatz import hardware_efficient

    nlayers = 1
    nqubits = 4
    nfermions = 2

    circuit = Circuit(4)
    hardware_efficient_ansatz = hardware_efficient.hea(nlayers, nqubits)
    circuit.add(hardware_efficient_ansatz)
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
    from qibo import Circuit

    from qibochem.driver.molecule import Molecule
    from qibochem.measurement.expectation import expectation
    from qibochem.ansatz import hardware_efficient

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74804))])
    mol.run_pyscf()
    hamiltonian = mol.hamiltonian()

    # Define and build the HEA
    nlayers = 1
    nqubits = mol.nso
    ntheta = 2 * nqubits * nlayers
    hea_ansatz = hardware_efficient.hea(nlayers, nqubits)

    circuit = Circuit(nqubits)
    circuit.add(hea_ansatz)

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


Unitary Coupled Cluster Ansatz
------------------------------

The Unitary Coupled Cluster (UCC) ansatz [#f1]_ [#f2]_ [#f3]_ is a variant of the popular gold standard Coupled Cluster ansatz [#f3]_ of quantum chemistry.
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
    from qibochem.ansatz.hf_reference import hf_circuit
    from qibochem.ansatz.ucc import ucc_circuit

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


.. rubric:: References

.. [#f1] Kutzelnigg, W. (1977). 'Pair Correlation Theories', in Schaefer, H.F. (eds) Methods of Electronic Structure Theory. Modern Theoretical Chemistry, vol 3. Springer, Boston, MA.

.. [#f2] Whitfield, J. D. et al., 'Simulation of electronic structure Hamiltonians using quantum computers', Mol. Phys. 109 (2011) 735.

.. [#f3] Anand. A. et al., 'A quantum computing view on unitary coupled cluster theory', Chem. Soc. Rev. 51 (2022) 1659.

.. [#f4] Crawford, T. D. et al., 'An Introduction to Coupled Cluster Theory for Computational Chemists', in Reviews in Computational Chemistry 14 (2007) 33.

.. [#f5] Barkoutsos, P. K. et al., 'Quantum algorithms for electronic structure calculations: Particle-hole Hamiltonian and optimized wave-function expansions', Phys. Rev. A 98 (2018) 022322.
