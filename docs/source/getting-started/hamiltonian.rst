Hamiltonian
===========

Central to quantum chemical calculations for molecules is the fermionic two-body Hamiltonian in second quantized form. 

TODO: spinfree, nonrelativistic, without external fields, without nuclear-repulsion energy

.. math::
    \hat{H} = \sum_{pq} h_{pq} a^\dagger_p a_q + \frac12 \sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_r a_s a_q 



The integrals :math:`h_{pq}` and :math:`h_{pqrs}` are one- and two-electron integrals in atomic units. For spinorbitals :math:`\phi_j` that make up the basis, the integrals are [#f1]_:

.. math:: 

    h_{pq} = \int \phi^*_p(\mathbf{x}_1)\left( -\frac12 \nabla^2 - \sum_A \frac{Z_A}{r_{A}} \right) \phi_q(\mathbf{x}_1) d\mathbf{x}_1

.. math:: 

    h_{pqrs} = \int \int \phi^*_p(\mathbf{x}_1)\phi^*_r(\mathbf{x}_2) \frac{1}{r_{12}} \phi_s(\mathbf{x}_2)\phi_q(\mathbf{x}_1) dx_1 dx_2

These integral quantities are obtained from the PySCF driver program. Qibochem then uses these integrals and OpenFermion to construct the second quantized fermionic Hamiltonian for the molecular system in terms of creation and annihilation operators, with coefficients from these integral quantities. 

.. code-block::

    mol = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))])
    mol.run_pyscf()
    mol_ferm_ham = mol.hamiltonian("ferm") # or mol.hamiltonian("f")


The fermionic Hamiltonian can then be mapped into qubit Hamiltonians in OpenFermion or in symbolic (:code:`sympy`) form. Additional information about the data structure of the second quantized fermionic Hamiltonian and qubit Hamiltonian can be found `here <https://quantumai.google/openfermion/tutorials/intro_to_openfermion>`_.

.. code-block::

    mol_qubit_ham = mol.hamiltonian("qubit") # or mol.hamiltonian("q")
    mol_sym_ham = mol.hamiltonian("sym")     # or mol.hamiltonian("s")

By default, the symbolic Hamiltonian is returned, i.e. if no arguments are given for :code:`mol.hamiltonian()`. 


Fermion to Qubit mapping
------------------------

Fermionic Hamiltonians have to be mapped to Qubit Hamiltonians in the Pauli basis for quantum computation. Supported mapping schemes are the Jordan-Wigner and Bravyi-Kitaev schemes, as implemented in OpenFermion. 

Jordan-Wigner
^^^^^^^^^^^^^

.. math:: 

    \hat{a}^\dagger_j = \bigotimes_{i=1}^{j-1} \hat{Z}_i \otimes (\hat{X}_j - i\hat{Y}_j) 
    
    
.. math:: 

    \hat{a}_j = \bigotimes_{i=1}^{j-1} \hat{Z}_i \otimes (\hat{X}_j + i\hat{Y}_j) 

This is done using the keyword :code:`ferm_qubit_map`.

.. code-block::

    mol_qubit_ham = mol.hamiltonian("qubit", ferm_qubit_map="jw")


Bravyi-Kitaev
^^^^^^^^^^^^^




.. rubric:: References

.. [#f1] Helgaker, T., Jørgensen, P., Olsen, J. Molecular Electronic Structure Theory 2000, Wiley, Chichester, England

