# Qibochem CLI examples

Each YAML below is a self-contained input for `qibochem run`.

| File | What it does | Replicates |
|---|---|---|
| `h2_hf.yaml` | Hartree-Fock on H2 / STO-3G | tutorial: molecule + hamiltonian |
| `h2_vqe_ucc.yaml` | UCCSD VQE on H2; recovers FCI | tutorial: ansatz |
| `lih_vqe_ucc.yaml` | UCCSD VQE on LiH with HF embedding active=[1,2,5] | `examples/ucc_example1.py` |
| `h3p_basis_rotation.yaml` | Basis-rotation ansatz on H3+ | `examples/br_example.py` |
| `pes_scan/run_scan.py` | H2 PES (potential energy surface) scan; writes a matplotlib PNG and CSV | (new) |

## Usage

```bash
qibochem run examples/cli/h2_vqe_ucc.yaml
# writes h2_vqe_ucc.result.json next to the yaml

qibochem template vqe > my_input.yaml
# edit my_input.yaml ...
qibochem run my_input.yaml

qibochem inspect examples/lih.xyz --basis sto-3g
# quick HF / qubit-count / Pauli-term-count summary

python examples/cli/pes_scan/run_scan.py
# writes h2_pes.png + h2_pes.csv into examples/cli/pes_scan/, with
# per-distance YAMLs + result JSONs preserved under pes_scan/_runs/
```
