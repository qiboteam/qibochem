Code overview
=============

Qibochem acts as a plug-in to Qibo for running chemistry-related simulations on quantum computers, either simulated classically or on actual quantum hardware.

Features
--------

The main objects in Qibochem are the `Molecule` class, defined in ``qibochem/driver/molecule.py``, and ansatzes, defined in ``qibochem/ansatzes``.
These allow a user to set up a Qibo circuit to run a VQE simulation to find the electronic energy of the molecular system of interest.
