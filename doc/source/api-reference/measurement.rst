===========
Measurement
===========

This section covers the API reference for obtaining the expectation value of a (molecular) Hamiltonian, either from a
state vector simulation, or from sample measurements.

Expectation value of Hamiltonian
--------------------------------

.. autofunction:: qibochem.measurement.result.expectation

.. _expectation-samples:

.. autofunction:: qibochem.measurement.result.expectation_from_samples

Measurement cost reduction
--------------------------

The following functions are used for reducing and optimising the measurement cost of obtaining the Hamiltonian
expectation value using sample measurements instead of a state vector simulation.

.. autofunction:: qibochem.measurement.optimization.measurement_basis_rotations

.. autofunction:: qibochem.measurement.optimization.allocate_shots
