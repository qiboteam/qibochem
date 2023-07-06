Hamiltonian mapping
===================

Central to quantum chemical calculations for molecules is the fermionic two-body Hamiltonian in second quantized form.

.. math::
    \hat{H} = \sum_{pq} h_{pq} a^\dagger_p a_q + \frac12 \sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_s a_r

which is to mapped to a qubit Hamiltonian for calculation of properties such as molecular energies using quantum hardware. 

The integrals :math:`h_{pq}` and :math:`h_{pqrs}` are one- and two-electron integrals. For spinorbitals :math:`\phi_j` that make up the basis, the integrals are:

.. math:: 
    h_{pq} = \int \phi^*_p(x_1)\hat{h}\phi_q(x_1) dx_1

.. math:: 
    h_{pqrs} = \int \int \phi^*_p(x_1)\phi^*_q(x_2)\hat{g}\phi_r(x_2)\phi_s(x_1) dx_1 dx_2

Supported mappings are the Jordan-Wigner and Bravyi-Kitaev mapping, as implemented in OpenFermion. 

Jordan-Wigner
-------------

.. math:: 
    \hat{a}^\dagger_j = \bigotimes_{i=1}^{j-1} \hat{Z}_i \otimes (\hat{X}_j - i\hat{Y}_j) 
    
    
.. math:: 
    \hat{a}_j = \bigotimes_{i=1}^{j-1} \hat{Z}_i \otimes (\hat{X}_j + i\hat{Y}_j) 




