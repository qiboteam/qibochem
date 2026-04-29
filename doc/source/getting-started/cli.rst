.. _cli:

Command-line interface
======================

Qibochem ships with a ``qibochem`` command that drives quantum-chemistry
workflows from a single YAML file. It's a thin wrapper over the same Python API
used in notebooks; if a workflow can be expressed as a notebook cell, it can
usually be expressed as a YAML.

Quickstart
----------

After installing Qibochem (``pip install qibochem`` or ``poetry install``):

.. code-block:: bash

    # Generate a starter input
    qibochem template vqe > my_input.yaml

    # Edit my_input.yaml to taste, then run:
    qibochem run my_input.yaml

    # Quick sanity check on a geometry, no YAML needed:
    qibochem inspect path/to/molecule.xyz --basis sto-3g

    # Validate without running (CI-friendly):
    qibochem validate my_input.yaml

A successful ``qibochem run`` writes ``<input>.result.json`` next to the input
file with the energies, timings, the full resolved config, the input file's
SHA-256, the qibochem version, and the git commit (if available) — enough to
reproduce the run.

Subcommands
-----------

``qibochem run <yaml>``
    Validate, build, and execute. Pass ``--dry-run`` to skip optimisation and
    only build/inspect the molecule + Hamiltonian.

``qibochem validate <yaml>``
    Schema-check only; no PySCF, no circuit construction.

``qibochem template <hf|vqe>``
    Print a heavily commented starter YAML to stdout.

``qibochem inspect <xyz>``
    PySCF-run a geometry and report ``e_hf``, qubit count, and Pauli term
    count. Quick cost estimate before launching a long VQE run.

YAML schema (v1)
----------------

The top level always has five blocks: ``version``, ``molecule``,
``hamiltonian``, ``method``, ``output``.

.. code-block:: yaml

    version: 1

    molecule:
      # Exactly one of:
      xyz_file: lih.xyz                # path is resolved relative to THIS yaml
      # geometry:
      #   - [Li, [0.0, 0.0, 0.0]]
      #   - [H,  [0.0, 0.0, 1.6]]
      charge: 0
      multiplicity: 1
      basis: sto-3g
      active: null                     # or [1, 2, 5] for HF embedding

    hamiltonian:
      mapping: jw                      # jw | bk

    method:
      kind: vqe                        # hf | vqe  (qsci | qse coming via selected_ci)
      # ... method-specific keys (see below)

    output:
      dir: null                        # default: same directory as the yaml
      prefix: null                     # default: yaml filename stem
      save_circuit: false              # write OpenQASM alongside the JSON result
      save_parameters: true
      log_level: info                  # debug | info | warning | error

Method block: HF
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    method:
      kind: hf

Reports the Hartree-Fock energy from PySCF — no quantum circuit involved.
Useful as a sanity check or for comparing against other methods.

Method block: VQE
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    method:
      kind: vqe
      ansatz:
        kind: ucc                      # ucc | givens | he | qeb | symm | basis_rotation
        excitations: [singles, doubles]
        layers: 2                      # only used by kind=he
      optimizer:
        name: BFGS                     # any scipy.optimize.minimize method, or 'qibo'
        options: {maxiter: 200}
        initial_parameters: zeros      # zeros | random | mp2 | [list of floats]
        seed: 0
      n_shots: null                    # null = exact statevector

The ``mp2`` initial-parameter strategy is only available for ``ucc`` and
``givens`` ansatzes. Most ansatzes only support the JW mapping right now.

Replicating notebook examples
-----------------------------

The ``examples/cli/`` folder contains YAMLs that mirror the existing Python
example scripts and tutorial workflows:

- ``examples/cli/h2_hf.yaml`` — the molecule/hamiltonian tutorial
- ``examples/cli/h2_vqe_ucc.yaml`` — UCCSD on H2
- ``examples/cli/lih_vqe_ucc.yaml`` — replicates ``examples/ucc_example1.py``
- ``examples/cli/h3p_basis_rotation.yaml`` — replicates ``examples/br_example.py``
- ``examples/cli/pes_scan/run_scan.py`` — H2 potential-energy-surface scan;
  loops the CLI over bond lengths and writes a matplotlib PNG + CSV
  alongside the per-distance YAMLs and result JSONs (in ``_runs/``)
