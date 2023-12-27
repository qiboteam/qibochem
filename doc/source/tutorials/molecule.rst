Building a Molecule
===================

The ``Molecule`` object contains all the information from a classical quantum chemistry calculation that will be used in the quantum simulation.
The molecular geometry is the main requirement for initialising a ``Molecule``, and can be either given inline, or using an ``.xyz`` file.

An example for a H\ :sub:`2`\  molecule with H-H distance of 0.74804 Angstroms:

.. code-block::

    from qibochem.driver.molecule import Molecule

    # Inline definition
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])

    # From an .xyz file
    # h2 = Molecule(xyz_file='h2.xyz')

The format of the ``xyz_file`` follows the usual convention:

  .. code-block::

    n_atoms

    atom_1 x_coord y_coord z_coord
    ...
    atom_n x_coord y_coord z_coord

The comment line in the ``.xyz`` file can be used to define the electronic charge and spin multiplicity of the molecule as well.
If the electronic charge and spin multiplicity of the molecule are not given (either in-line or in the ``xyz_file``), they default to 0 and 1 respectively.
Additional details are provided in the :ref:`API reference <molecule-class>`.


Obtaining the molecular integrals
---------------------------------

After defining the molecular coordinates, the next step is to obtain the one- and two- electron integrals.
Qibochem offers the functionality to interface with `PySCF <https://pyscf.org/>`_ towards that end.

.. code-block:: python

    from qibochem.driver.molecule import Molecule

    # Inline definition of H2
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])
    # Running PySCF
    h2.run_pyscf()

The default level of theory is HF/STO-3G, and upon executing the PySCF driver for a given molecule, several molecular quantities are calculated and stored in the Molecule class.
These include the:

* converged Hartree-Fock energy
* optimized molecular orbital (MO) coefficients
* one- and two-electron integrals
