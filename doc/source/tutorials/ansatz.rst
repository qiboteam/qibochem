
.. _Ansatz tutorial:

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

    from qibochem.ansatz import he_circuit

    nqubits = 4
    nlayers = 1

    circuit = he_circuit(nqubits, nlayers)
    circuit.draw()

.. code-block:: output

    0: ─RY─RZ─o─────Z─
    1: ─RY─RZ─Z─o───|─
    2: ─RY─RZ───Z─o─|─
    3: ─RY─RZ─────Z─o─

The energy of the state generated from the hardware efficient ansatz for the fermionic two-body Hamiltonian can then be estimated, using state vectors or samples.

The following example demonstrates how the energy of the H\ :sub:`2` molecule is affected with respect to the rotational parameters:

.. code-block:: python

    import numpy as np

    from qibochem.driver import Molecule
    from qibochem.measurement import expectation
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

An example of how to build a UCC doubles circuit ansatz for the H\ :sub:`2` molecule is given as:

.. code-block:: python

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit, ucc_circuit

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74804))])
    mol.run_pyscf()
    hamiltonian = mol.hamiltonian()

    # Set parameters for the rest of the experiment
    nqubits = mol.nso
    nelectrons = mol.nelec

    # Build UCCD circuit
    circuit = hf_circuit(nqubits, nelectrons) # Start with HF circuit
    circuit += ucc_circuit(nqubits, [0, 1, 2, 3]) # Then add the double excitation circuit ansatz

    circuit.draw()

.. code-block:: output

    0:     ─X───H─────X─RZ─X─────H─SDG─H─────────X─RZ─X─────H─S─SDG─H─────X─RZ─X─ ...
    1:     ─X───H───X─o────o─X───H─SDG─H───────X─o────o─X───H─S─H───────X─o────o─ ...
    2:     ─SDG─H─X─o────────o─X─H─S───SDG─H─X─o────────o─X─H─S─H─────X─o──────── ...
    3:     ─H─────o────────────o─H─H─────────o────────────o─H─H───────o────────── ...

    0: ... ────H─S───H─────X─RZ─X─────H─SDG─H─────X─RZ─X─────H─S───H─────────X─RZ ...
    1: ... X───H─SDG─H───X─o────o─X───H─S───H───X─o────o─X───H─SDG─H───────X─o─── ...
    2: ... o─X─H─H─────X─o────────o─X─H─SDG─H─X─o────────o─X─H─S───SDG─H─X─o───── ...
    3: ... ──o─H─H─────o────────────o─H─SDG─H─o────────────o─H─S───SDG─H─o─────── ...

    0: ... ─X─────H─H───────────X─RZ─X─────H─SDG─H─────────X─RZ─X─────H─S─
    1: ... ─o─X───H─S─H───────X─o────o─X───H─SDG─H───────X─o────o─X───H─S─
    2: ... ───o─X─H─S─H─────X─o────────o─X─H─H─────────X─o────────o─X─H───
    3: ... ─────o─H─S─SDG─H─o────────────o─H─S───SDG─H─o────────────o─H─S─


UCC with Qubit-Excitation-Based :math:`n`-tuple Excitation
----------------------------------------------------------

A CNOT depth-efficient quantum circuit for employing the UCC ansatz, dubbed the Qubit-Excitation-Based (QEB) n-tuple excitations for UCC, was constructed by Yordanov et al. [#f6]_ and Magoulas et al. [#f7]_, avoiding the exponential number of CNOT cascades in those developed before. [#f5]_ The quantum circuits generated for :math:`N` qubits have a reduction of CNOTs from :math:`(2N-1)2^{2N}` to :math:`2^{2N-1}+4N-2`.

An example for the H\ :sub:`2` molecule is given here:


.. code-block:: python

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit, qeb_circuit

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74804))])
    mol.run_pyscf()
    hamiltonian = mol.hamiltonian()

    # Set parameters for the rest of the experiment
    n_qubits = mol.nso
    n_electrons = mol.nelec

    # Build UCCD circuit
    circuit = hf_circuit(n_qubits, n_electrons) # Start with HF circuit
    circuit += qeb_circuit(n_qubits, [0, 1, 2, 3]) # Then add the double excitation circuit ansatz

    circuit.draw()

.. code-block:: output

    0: ─X─X─────X─o──X─────X─
    1: ─X─o───X───o────X───o─
    2: ─────X─|─X─o──X─|─X───
    3: ─────o─o───RY───o─o───

..
   _Basis rotation ansatz

Basis rotation ansatz
---------------------

The starting points for contemporary quantum chemistry methods are often those based on the mean field approximation within a (finite) molecular orbital basis, i.e. the Hartree-Fock method. The electronic energy is calculated as the mean value of the electronic Hamiltonian :math:`\hat{H}_{\mathrm{elec}}` acting on a normalized single Slater determinant function :math:`\psi` [#f8]_

.. math::

    \begin{align*}
    E[\psi] &= \langle \psi | \hat{H}_{\mathrm{elec}} |\psi \rangle \\
            &= \sum_i^{N_f} \langle \phi_i |\hat{h}|\phi_i \rangle + \frac{1}{2} \sum_{i,j}^{N_f}
            \langle \phi_i\phi_j||\phi_i\phi_j \rangle
    \end{align*}

The orthonormal molecular orbitals :math:`\phi` are optimized by a direct minimization of the energy functional with respect to parameters :math:`\kappa` that parameterize the unitary rotations of the orbital basis. Qibochem's implementation uses the QR decomposition of the unitary matrix as employed by Clements et al., [#f9]_ which results in a rectangular gate layout of `Givens rotation gates <https://qibo.science/qibo/stable/api-reference/qibo.html#givens-gate>`_ that yield linear CNOT gate depth when decomposed.


.. code-block:: python

    from qibo.models import VQE

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit, basis_rotation_circuit

    h3p = Molecule(
        [
            ("H", (0.0000, 0.0000, 0.0000)),
            ("H", (0.0000, 0.0000, 0.8000)),
            ("H", (0.0000, 0.0000, 1.6000)),
        ],
        charge=1,
        multiplicity=1,
    )
    h3p.run_pyscf(max_scf_cycles=1)

    e_init = h3p.e_hf
    h3p_sym_ham = h3p.hamiltonian("sym", h3p.oei, h3p.tei, 0.0, "jw")

    circuit = basis_rotation_circuit(h3p.nso, h3p.nelec, parameters=0.1)
    circuit_parameters = [param for _param in circuit.get_parameters() for param in _param]  # Flatten list

    circuit.draw()

    vqe = VQE(circuit, h3p_sym_ham)
    result = vqe.minimize(circuit_parameters)

    print("energy of initial guess: ", e_init)
    print("energy after vqe       : ", result[0])

.. code-block:: output

    basis rotation: using uniform value of 0.1 for each parameter value
    0: ─X─G─────────G─────────G─────────
    1: ─X─G─────G───G─────G───G─────G───
    2: ─────G───G─────G───G─────G───G───
    3: ─────G─────G───G─────G───G─────G─
    4: ───────G───G─────G───G─────G───G─
    5: ───────G─────────G─────────G─────
    energy of initial guess:  -1.1977713400022745
    energy after vqe       :  -1.2025110984672576


.. rubric:: References

.. [#f1] Kutzelnigg, W. (1977). 'Pair Correlation Theories', in Schaefer, H.F. (eds) Methods of Electronic Structure Theory. Modern Theoretical Chemistry, vol 3. Springer, Boston, MA.

.. [#f2] Whitfield, J. D. et al., 'Simulation of Electronic Structure Hamiltonians using Quantum Computers', Mol. Phys. 109 (2011) 735.

.. [#f3] Anand. A. et al., 'A Quantum Computing view on Unitary Coupled Cluster Theory', Chem. Soc. Rev. 51 (2022) 1659.

.. [#f4] Crawford, T. D. et al., 'An Introduction to Coupled Cluster Theory for Computational Chemists', in Reviews in Computational Chemistry 14 (2007) 33.

.. [#f5] Barkoutsos, P. K. et al., 'Quantum algorithms for electronic structure calculations: Particle-hole Hamiltonian and optimized wave-function expansions', Phys. Rev. A 98 (2018) 022322.

.. [#f6] Yordanov Y. S. et al., 'Efficient Quantum Circuits for Quantum Computational Chemistry', Phys Rev A 102 (2020) 062612.

.. [#f7] Magoulas, I. and Evangelista, F. A., 'CNOT-Efficient Circuits for Arbitrary Rank Many-Body Fermionic and Qubit Excitations', J. Chem. Theory Comput. 19 (2023) 822.

.. [#f8] Piela, L. (2007). 'Ideas of Quantum Chemistry'. Elsevier B. V., the Netherlands.

.. [#f9] Clements, W. R. et al., 'Optimal Design for Universal Multiport Interferometers', Optica 3 (2016) 1460.
