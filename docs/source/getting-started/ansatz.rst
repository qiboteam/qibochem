======
Ansatz
======

A quantum circuit consisting of parameterized gates RX(theta), RY(theta) and RZ(theta), represents a unitary transformation :math:`U(\theta)` that transforms the initial quantum state into an Ansatz state :math:`|\psi(\theta)\rangle`.

Hardware Efficient Ansatz
-------------------------

Qibochem provides a hardware efficient ansatz that simply consists of a layer of single qubit rotation gates followed by a layer of two-qubit gates that entangle the qubits. For the H\ :sub:`2` case discussed in previous sections, a possible hardware efficient ansatz can be constructed as such:

.. image:: qibochem_doc_ansatz_hardware-efficient.svg

Example
^^^^^^^

Placeholder for hardware-efficient ansatz example


Unitary Coupled Cluster Ansatz
------------------------------

The Unitary Coupled Cluster ansatz [#f1]_ [#f2]_ is a variant of the popular gold standard Coupled Cluster ansatz [#f3]_ of quantum chemistry. The unitary coupled cluster wave function is a parameterized unitary transformation of a reference wave function, of which a common choice is the Hartree-Fock wave function.

.. math::

    |\psi_{\mathrm{UCC}}\rangle = U(\theta)|\psi_{\mathrm{ref}}\rangle

Example
^^^^^^^

Placeholder for UCCD example

.. rubric:: References

.. [#f1] Whitfield, J. D. et al., 'Simulation of electronic structure Hamiltonians using quantum computers', Mol. Phys. 109 (2011) 735.

.. [#f2] Anand. A. et al., 'A quantum computing view on unitary coupled cluster theory', Chem. Soc. Rev. 51 (2022) 1659.

.. [#f3] Crawford, T. D. et al., 'An Introduction to Coupled Cluster Theory for Computational Chemists', in Reviews in Computational Chemistry 14 (2007) 33.
