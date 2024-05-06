Molecular Hamiltonian
=====================


Central to quantum chemical calculations for molecules is the fermionic spin-free nonrelativistic two-body Hamiltonian in second quantized form in the absence of external fields.
Under the Born-Oppenheimer approximation, the nuclear repulsion energy can be taken to be constant, which allows us to write the molecular Hamiltonian as:

.. math::
    \hat{H} = \sum_{pq} h_{pq} a^\dagger_p a_q + \frac12 \sum_{pqrs} g_{pqrs} a^\dagger_p a^\dagger_r a_s a_q

The integrals :math:`h_{pq}` and :math:`g_{pqrs}` are the one- and two-electron integrals in atomic units. For spin-orbitals :math:`\phi_j` that make up the basis, the integrals are [#f1]_:

.. math::

    h_{pq} = \int \phi^*_p(\mathbf{x}_1)\left( -\frac12 \nabla^2 - \sum_A \frac{Z_A}{r_{A}} \right) \phi_q(\mathbf{x}_1) d\mathbf{x}_1

.. math::

    g_{pqrs} = \int \int \phi^*_p(\mathbf{x}_1)\phi^*_r(\mathbf{x}_2) \frac{1}{r_{12}} \phi_s(\mathbf{x}_2)\phi_q(\mathbf{x}_1) dx_1 dx_2

Fermionic Hamiltonian
---------------------

The process for obtaining the molecular integrals using PySCF has been given in an earlier example.
Qibochem then uses these integrals and the `OpenFermion <https://quantumai.google/openfermion>`_ package to construct the second quantized fermionic Hamiltonian for the molecular system in terms of creation and annihilation operators.

.. code-block:: python

    from qibochem.driver.molecule import Molecule

    mol = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])
    mol.run_pyscf()
    mol_ferm_ham = mol.hamiltonian("ferm") # or mol.hamiltonian("f")
    print(mol_ferm_ham.terms)

Output:

.. code-block:: output

    {(): 0.7074183344740923, ((0, 1), (0, 0)): -1.248461959132892, ((1, 1), (1, 0)): -1.248461959132892, ((2, 1), (2, 0)): -0.48007161818330846, ((3, 1), (3, 0)): -0.48007161818330846, ((0, 1), (0, 1), (0, 0), (0, 0)): 0.3366109237586995, ((0, 1), (0, 1), (2, 0), (2, 0)): 0.09083064962340165, ((0, 1), (1, 1), (1, 0), (0, 0)): 0.3366109237586995, ((0, 1), (1, 1), (3, 0), (2, 0)): 0.09083064962340165, ((0, 1), (2, 1), (0, 0), (2, 0)): 0.09083064962340165, ((0, 1), (2, 1), (2, 0), (0, 0)): 0.33115823068165495, ((0, 1), (3, 1), (1, 0), (2, 0)): 0.09083064962340165, ((0, 1), (3, 1), (3, 0), (0, 0)): 0.33115823068165495, ((1, 1), (0, 1), (0, 0), (1, 0)): 0.3366109237586995, ((1, 1), (0, 1), (2, 0), (3, 0)): 0.09083064962340165, ((1, 1), (1, 1), (1, 0), (1, 0)): 0.3366109237586995, ((1, 1), (1, 1), (3, 0), (3, 0)): 0.09083064962340165, ((1, 1), (2, 1), (0, 0), (3, 0)): 0.09083064962340165, ((1, 1), (2, 1), (2, 0), (1, 0)): 0.33115823068165495, ((1, 1), (3, 1), (1, 0), (3, 0)): 0.09083064962340165, ((1, 1), (3, 1), (3, 0), (1, 0)): 0.33115823068165495, ((2, 1), (0, 1), (0, 0), (2, 0)): 0.3311582306816552, ((2, 1), (0, 1), (2, 0), (0, 0)): 0.09083064962340165, ((2, 1), (1, 1), (1, 0), (2, 0)): 0.3311582306816552, ((2, 1), (1, 1), (3, 0), (0, 0)): 0.09083064962340165, ((2, 1), (2, 1), (0, 0), (0, 0)): 0.09083064962340165, ((2, 1), (2, 1), (2, 0), (2, 0)): 0.348087115228365, ((2, 1), (3, 1), (1, 0), (0, 0)): 0.09083064962340165, ((2, 1), (3, 1), (3, 0), (2, 0)): 0.348087115228365, ((3, 1), (0, 1), (0, 0), (3, 0)): 0.3311582306816552, ((3, 1), (0, 1), (2, 0), (1, 0)): 0.09083064962340165, ((3, 1), (1, 1), (1, 0), (3, 0)): 0.3311582306816552, ((3, 1), (1, 1), (3, 0), (1, 0)): 0.09083064962340165, ((3, 1), (2, 1), (0, 0), (1, 0)): 0.09083064962340165, ((3, 1), (2, 1), (2, 0), (3, 0)): 0.348087115228365, ((3, 1), (3, 1), (1, 0), (1, 0)): 0.09083064962340165, ((3, 1), (3, 1), (3, 0), (3, 0)): 0.348087115228365}


Fermion to Qubit mapping
------------------------

Next, in order to run chemistry simulations, the fermionic Hamiltonian must first be mapped onto a qubit Hamiltonian.
Qibochem supports the Jordan-Wigner and Bravyi-Kitaev mapping schemes (as implemented in OpenFermion),
and this can be specified using the keyword argument :code:`ferm_qubit_map` as follows:

.. code-block:: python

    mol_qubit_ham = mol.hamiltonian("qubit", ferm_qubit_map="jw") # or just mol.hamiltonian("q")
    print(mol_qubit_ham.terms)

Output:

.. code-block:: output

    {(): -0.10728041160866736, ((0, 'Z'),): 0.17018261181714206, ((1, 'Z'),): 0.17018261181714206, ((2, 'Z'),): -0.21975065439248248, ((3, 'Z'),): -0.21975065439248248, ((0, 'Z'), (1, 'Z')): 0.16830546187934975, ((0, 'Z'), (2, 'Z')): 0.1201637905291267, ((0, 'Z'), (3, 'Z')): 0.16557911534082753, ((1, 'Z'), (2, 'Z')): 0.16557911534082753, ((1, 'Z'), (3, 'Z')): 0.1201637905291267, ((2, 'Z'), (3, 'Z')): 0.1740435576141825, ((0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')): -0.045415324811700825, ((0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')): 0.045415324811700825, ((0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')): 0.045415324811700825, ((0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')): -0.045415324811700825}


Qibo SymbolicHamiltonian
------------------------

Lastly, to carry out quantum simulations of the molecular electronic structure using Qibo, the qubit Hamiltonian can be returned as a Qibo ``SymbolicHamiltonian``:

.. code-block:: python

    mol_sym_ham = mol.hamiltonian("sym")     # or mol.hamiltonian("s")
    print(mol_sym_ham.form)

.. code-block:: output

    -0.107280411608667 - 0.0454153248117008*X0*X1*Y2*Y3 + 0.0454153248117008*X0*Y1*Y2*X3 + 0.0454153248117008*Y0*X1*X2*Y3 - 0.0454153248117008*Y0*Y1*X2*X3 + 0.170182611817142*Z0 + 0.16830546187935*Z0*Z1 + 0.120163790529127*Z0*Z2 + 0.165579115340828*Z0*Z3 + 0.170182611817142*Z1 + 0.165579115340828*Z1*Z2 + 0.120163790529127*Z1*Z3 - 0.219750654392482*Z2 + 0.174043557614182*Z2*Z3 - 0.219750654392482*Z3

By default, the molecular Hamiltonian is returned as a ``SymbolicHamiltonian``, `i.e.` if no arguments are given in :code:`mol.hamiltonian()`.
In addition, if HF embedding has been applied, the embedded values of the one-/two- electron integrals will be used to construct the molecular Hamiltonian as well.

Otherwise, using the ``"f"``/``"ferm"`` and ``"q"``/``"qubit"`` arguments will return the molecular Hamiltonian as an OpenFermion ``FermionOperator`` and ``QubitOperator`` respectively.
Additional information about the data structure of these two classes can be found `here <https://quantumai.google/openfermion/tutorials/intro_to_openfermion>`_.



.. rubric:: References

.. [#f1] Helgaker, T., Jørgensen, P., Olsen, J. Molecular Electronic Structure Theory 2000, Wiley, Chichester, England
