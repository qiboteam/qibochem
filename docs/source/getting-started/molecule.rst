Molecule
========

The Molecule object contains all the information from a classical quantum chemistry calculation that will be used in the quantum simulation. 

The key inputs required are:

geometry
  atomic coordinates in the form of a list of tuples

charge
  an integer value for the net charge for the entire molecule

  default value is 0

multiplicity
  spin multiplicity given as 2S + 1, where S is the number of unpaired electrons multipled by 1/2
  
  default value is 1 (all electrons paired)

Optional inputs:

basis
  the atomic orbital basis set to be used

  default value is 'STO-3G'

xyz_file
  qibochem allows the geometry, charge, and multiplicity to be specified as an .xyz file, with the format:

  .. code-block::
    
    n_atoms
    charge multiplicity
    atom_1 x_coord y_coord z_coord
    ...
    atom_n x_coord y_coord z_coord


active
  list of molecular orbitals to be considered in the quantum simulation

The current implementation supports both restricted and unrestricted spins. (TODO: elaborate)


Variables
---------

Upon executing the PyScf driver program for a given molecule, molecular quantities are calculated at the Hartree-Fock (HF) level and stored in the Molecule class. These include:

* converged Hartree-Fock energy
* optimized molecular orbital (MO) coefficients
* one- and two-electron integrals

Additional details are provided in the :ref:`API reference<api driver molecule class>`. 


Example
-------

The following examples show how to use the driver program to save molecule quantities into the Molecule object

PySCF for H\ :sub:`2`\  with H-H distance of 0.74804 Angstroms at HF/STO-3G level

.. code-block:: python

    from qibochem.driver.molecule import Molecule

    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74804))])
    h2.run_pyscf()
    print(h2.e_hf)

Output: 

.. code-block:: output 

    converged SCF energy = -1.11628373627429
    

    

