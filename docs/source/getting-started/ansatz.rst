======
Ansatz
======

A quantum circuit consisting of parameterized gates RX(theta), RY(theta) and RZ(theta), represents a unitary transformation :math:`U(\theta)` that transforms the initial quantum state into an Ansatz state :math:`|\psi(\theta)\rangle`. 

Hardware Efficient Ansatz
-------------------------

Qibochem provides a hardware efficient ansatz that simply consists of a layer of single qubit rotation gates followed by a layer of two-qubit gates that entangle the qubits. 

Example
^^^^^^^

Unitary Coupled Cluster Ansatz
------------------------------

The Unitary Coupled Cluster ansatz [#f1]_ [#f2]_

.. rubric:: References

.. [#f1] Whitfield, J. D. et al., 'Simulation of electronic structure Hamiltonians using quantum computers', Mol. Phys. 109 (2011) 735.

.. [#f2] Anand. A. et al., 'A quantum computing view on unitary coupled cluster theory', Chem. Soc. Rev. 51 (2022) 1659. 