======
Ansatz
======

This section covers the API reference for various chemistry circuit ansatzes.

Hardware-efficient
------------------

.. autofunction:: qibochem.ansatz.ansatz.he_circuit


Hartree-Fock
------------

.. autofunction:: qibochem.ansatz.ansatz.hf_circuit


Unitary Coupled Cluster
-----------------------

.. autofunction:: qibochem.ansatz.ansatz.ucc_circuit

.. Comment out for now
  .. autofunction:: qibochem.ansatz.ucc.ucc_ansatz

.. autofunction:: qibochem.ansatz.ansatz.qeb_circuit


Basis rotation
--------------

.. autofunction:: qibochem.ansatz.basis_rotation.basis_rotation_gates

Givens Excitation
-----------------

.. autofunction:: qibochem.ansatz.ansatz.givens_circuit

.. Comment out for now
  .. autofunction:: qibochem.ansatz.givens_excitation.givens_excitation_ansatz

Symmetry-Preserving
-------------------

.. autofunction:: qibochem.ansatz.symmetry.symm_preserving_circuit
