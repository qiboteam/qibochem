Building a Molecule
===================

To get started with Qibochem, the molecular system of interest is first defined:

Molecular geometry input
------------------------

A ``Molecule`` can be defined either inline, or using an ``.xyz`` file. An example for a H2 molecule:

.. code-block::

    from qibochem.driver.molecule import Molecule
    
    # Inline definition
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])

    # From an .xyz file
    # h2 = Molecule(xyz_file='h2.xyz')

Note that the comment line in the ``.xyz`` file can be used to define the electronic charge and spin multiplicity of the molecule.
If it is not given, it defaults to 0 and 1, respectively.


Obtaining the molecular Hamiltonian
-----------------------------------

After defining the molecular coordinates, the next step is to obtain the molecular integrals.
Qibochem offers the functionality to interface with either `PySCF`_ or `PSI4`_ towards that end:

.. _PySCF: https://pyscf.org/
.. _PSI4: https://psicode.org/

.. code-block::

    from qibochem.driver.molecule import Molecule
    
    # Inline definition of H2
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])

    # Using PySCF
    h2.run_pyscf()
    # Using PSI4
    # h2.run_psi4()

After the molecular integrals have been calculated, molecular Hamiltonian can then be constructed in the form of a Qibo ``SymbolicHamiltonian``:

.. code-block::

    from qibochem.driver.molecule import Molecule
    
    # Inline definition of H2
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])

    # Calculate molecular integrals
    h2.run_pyscf()

    # Get molecular Hamiltonian
    hamiltonian = h2.hamiltonian()

