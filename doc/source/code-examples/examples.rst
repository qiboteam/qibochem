Basic examples
==============

Here are a few short basic `how to` examples.

How to define a Molecule in Qibochem
------------------------------------

A ``Molecule`` can be defined either inline, or using an ``.xyz`` file. An example for a H2 molecule:

.. code-block::

    from qibochem.driver.molecule import Molecule
    
    # Inline definition
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])

    # From an .xyz file
    # h2 = Molecule(xyz_file='h2.xyz')

Interfacing with PSI4/PySCF to obtain the 1-/2- electron integrals
------------------------------------------------------------------

After defining a ``Molecule``, the next step is to obtain the molecular integrals. Qibochem has has functions that interface with either `PySCF`_ or `PSI4`_ to do so:

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

Next section
------------

hello world


