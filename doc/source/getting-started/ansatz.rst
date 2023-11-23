======
Ansatz
======

A quantum circuit consisting of parameterized gates RX(theta), RY(theta) and RZ(theta), represents a unitary transformation :math:`U(\theta)` that transforms the initial quantum state into an Ansatz state :math:`|\psi(\theta)\rangle`.

Hardware Efficient Ansatz
-------------------------

Qibochem provides a hardware efficient ansatz that simply consists of a layer of single-qubit rotation gates followed by a layer of two-qubit gates that entangle the qubits. For the H\ :sub:`2` case discussed in previous sections, a possible hardware efficient ansatz can be constructed as such:

.. image:: qibochem_doc_ansatz_hardware-efficient.svg

Example
^^^^^^^

.. code-block:: python

    from qibochem.ansatz import hardware_efficient
    import qibo

    nlayers = 1
    nqubits = 4
    nfermions = 2

    circuit = qibo.models.Circuit(4)
    circuit.add(gates.X(_i) for _i in range(nfermions))
    hardware_efficient_ansatz = hardware_efficient.hea(nlayers, nqubits)
    circuit.add(hardware_efficient_ansatz)
    print(circuit.draw())

.. code-block:: output

    q0: ─X──RY─RZ─o─────Z─
    q1: ─X──RY─RZ─Z─o───|─
    q2: ─RY─RZ──────Z─o─|─
    q3: ─RY─RZ────────Z─o─

The energy of the state generated from the hardware efficient ansatz for the fermionic two-body Hamiltonian can then be estimated, using statevectors or samples. The following example calculates the energy of the H2 molecule.

Example 
^^^^^^^

.. code-block:: python 

    from qibochem.driver.molecule import Molecule
    from qibochem.measurement.expectation import expectation
    from qibochem.ansatz import hardware_efficient
    import numpy as np
    from qibo import models, gates

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74804))])
    mol.run_pyscf()
    mol_classical_hf_energy = mol.e_hf
    mol_sym_ham = mol.hamiltonian("s")

    nlayers = 1
    nqubits = mol.nso
    ntheta = 2 * nqubits * nlayers
    theta = np.zeros(ntheta)

    hea_ansatz = hardware_efficient.hea(nlayers, nqubits)
    circuit = models.Circuit(nqubits)
    circuit.add(gates.X(_i) for _i in range(sum(mol.nelec)))
    circuit.add(hea_ansatz)
    circuit.set_parameters(theta)

    hf_energy = expectation(circuit, mol_sym_ham)

    print('classical HF/STO-3G energy for H2 at 0.74804 Angstroms: ', mol_classical_hf_energy)
    print('quantum   HF/STO-3G energy for H2 at 0.74804 Angstroms: ', hf_energy)


.. code-block:: output



Unitary Coupled Cluster Ansatz
------------------------------

The Unitary Coupled Cluster (UCC) ansatz [#f1]_ [#f2]_ [#f3]_ is a variant of the popular gold standard Coupled Cluster ansatz [#f3]_ of quantum chemistry. The UCC wave function is a parameterized unitary transformation of a reference wave function :math:`\psi_{\mathrm{ref}})`, of which a common choice is the Hartree-Fock wave function.

.. math::

    |\psi_{\mathrm{UCC}}\rangle = U(\theta)|\psi_{\mathrm{ref}}\rangle

Implementation of the UCC ansatz on quantum computers involve Suzuki-Trotter decompositions of exponentials of Pauli operators. 

Example
^^^^^^^

Placeholder for UCCD example

.. rubric:: References

.. [#f1] Kutzelnigg, W. (1977). Pair Correlation Theories. In: Schaefer, H.F. (eds) Methods of Electronic Structure Theory. Modern Theoretical Chemistry, vol 3. Springer, Boston, MA.

.. [#f2] Whitfield, J. D. et al., 'Simulation of electronic structure Hamiltonians using quantum computers', Mol. Phys. 109 (2011) 735.

.. [#f3] Anand. A. et al., 'A quantum computing view on unitary coupled cluster theory', Chem. Soc. Rev. 51 (2022) 1659.

.. [#f4] Crawford, T. D. et al., 'An Introduction to Coupled Cluster Theory for Computational Chemists', in Reviews in Computational Chemistry 14 (2007) 33.
