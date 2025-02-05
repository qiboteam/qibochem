Building a Molecule
===================

The ``Molecule`` object contains all the information from a classical quantum chemistry calculation that will be used in the quantum simulation.
The molecular geometry is the main requirement for initialising a ``Molecule``, and can be either given inline, or using an ``.xyz`` file.

An example for a H\ :sub:`2`\  molecule with H-H distance of 0.74804 Angstroms:

.. code-block:: python

    from qibochem.driver import Molecule

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

    from qibochem.driver import Molecule

    # Inline definition of H2
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])
    # Running PySCF
    h2.run_pyscf()

The default level of theory is HF/STO-3G, and upon executing the PySCF driver for a given molecule, several molecular quantities are calculated and stored in the ``Molecule`` class.
These include the:

* converged Hartree-Fock energy
* optimized molecular orbital (MO) coefficients
* one- and two-electron integrals


Embedding the quantum electronic structure calculation
------------------------------------------------------

In quantum chemistry, most *ab initio* calculations start with a Hartree-Fock (HF) calculation.
The obtained HF wave function is then used as a starting point to apply post-HF methods to improve upon the treatment of electron correlation.
An example of a post-HF method that can be run on a quantum computer is the :ref:`Unitary Coupled Cluster method <UCC Ansatz>`.
Unfortunately, the current level of quantum hardware are unable to run these methods for molecules that are too large.

One possible approach to reduce the hardware required is to embed the quantum simulation into a classically computed environment.
(see `Rossmanek et al. <https://doi.org/10.1063/5.0029536/>`_)
Essentially, the occupancy of the core *1s* orbitals for the heavy atoms in a molecule is effectively constant; the same might hold for higher virtual orbitals.
As such, these orbitals can be removed from the simulation of the molecule without there being a significant change in the molecular properties of interest.
The remaining orbitals left in the simulation are then known as the active space of the molecule.

An example of how to apply this using Qibochem for the LiH molecule is given below.

.. code-block:: python

    from qibochem.driver import Molecule

    # Inline definition of H2
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])
    # Running PySCF
    h2.run_pyscf()
    frozen = [0] # The electrons on the 1s orbital of Li in LiH are frozen - removed from the quantum simulation
    h2.hf_embedding(frozen=frozen)

The code block above will re-calculate the one- and two-electron integrals for the given active space, and store the result back in the ``Molecule`` class.

.. Note: Not sure if this is the best place to describe HF embedding? But where else would be good?
