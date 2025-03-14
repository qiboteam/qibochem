Quick start
-----------

To quickly install and run Qibochem, open a terminal with ``python >= 3.9`` and type:

.. code-block::

   pip install qibochem

Here is an example of building the UCCD ansatz with the H2 molecule to test your installation:

.. testcode::

    import numpy as np
    from qibo.models import VQE

    from qibochem.driver import Molecule
    from qibochem.ansatz import hf_circuit, ucc_circuit

    # Define the H2 molecule and obtain its 1-/2- electron integrals with PySCF
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    # Generate the molecular Hamiltonian
    hamiltonian = h2.hamiltonian()

    # Build a UCC circuit ansatz for running VQE
    circuit = hf_circuit(h2.nso, h2.nelec)
    circuit += ucc_circuit(h2.nso, [0, 1, 2, 3])

    # Create and run the VQE, starting with random initial parameters
    vqe = VQE(circuit, hamiltonian)

    initial_parameters = np.random.uniform(0.0, 2*np.pi, 8)
    best, params, extra = vqe.minimize(initial_parameters)

..
  TODO: Another example with measurements
