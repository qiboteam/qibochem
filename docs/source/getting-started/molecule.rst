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


The current implementation supports both restricted and unrestricted spins. (TODO: elaborate)


Variables
---------

Instance variables that can be stored in the Molecule class are:



Example
-------

The following examples show how to use the driver program to save molecule quantities into the Molecule object

PySCF for H\ :sub:`2`\  with H-H distance of 0.74 Angstroms at HF/STO-3G level

.. code-block:: python

    from qibochem.driver import Molecule

    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))])
    h2.run_pyscf()
    

