"""H2 potential-energy-surface (PES) scan driven by the qibochem CLI.

Generates a YAML for each bond distance into ``_runs/``, invokes
``qibochem run`` on each, then plots HF and VQE energies vs bond length
using matplotlib. All artifacts (per-distance YAMLs, result JSONs, CSV,
PNG) land alongside this script so they can be inspected and re-plotted
without re-running the scan.

Usage:
    python examples/cli/pes_scan/run_scan.py
    python examples/cli/pes_scan/run_scan.py --points 21 --rmax 3.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

YAML_TEMPLATE = """\
version: 1
molecule:
  geometry:
    - [H, [0.0, 0.0, 0.0]]
    - [H, [0.0, 0.0, {distance:.6f}]]
  basis: sto-3g
hamiltonian:
  mapping: jw
method:
  kind: vqe
  ansatz:
    kind: ucc
    excitations: [singles, doubles]
  optimizer:
    name: BFGS
    options: {{maxiter: 200}}
    initial_parameters: mp2
    seed: 0
output:
  log_level: warning
  save_parameters: false
"""

HERE = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description="H2 PES scan via qibochem CLI.")
    parser.add_argument("--rmin", type=float, default=0.5, help="Min bond length (Angstrom).")
    parser.add_argument("--rmax", type=float, default=2.5, help="Max bond length (Angstrom).")
    parser.add_argument("--points", type=int, default=11, help="Number of scan points.")
    parser.add_argument("--csv", type=Path, default=HERE / "h2_pes.csv", help="Output CSV path.")
    parser.add_argument("--fig", type=Path, default=HERE / "h2_pes.png", help="Output figure path.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting; just write data.")
    args = parser.parse_args()

    qibochem = shutil.which("qibochem")
    if qibochem is None:
        sys.stderr.write("error: 'qibochem' command not found on PATH; install the package first.\n")
        return 2

    workdir = HERE / "_runs"
    workdir.mkdir(exist_ok=True)
    distances = np.linspace(args.rmin, args.rmax, args.points)

    e_hf: list[float] = []
    e_vqe: list[float] = []
    for i, d in enumerate(distances):
        yaml_path = workdir / f"h2_{i:03d}.yaml"
        yaml_path.write_text(YAML_TEMPLATE.format(distance=float(d)))
        print(f"[{i + 1}/{args.points}] r = {d:.3f} A ...", flush=True)
        proc = subprocess.run([qibochem, "run", str(yaml_path)], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            sys.stderr.write(f"  FAILED:\n{proc.stderr}\n")
            return proc.returncode
        result_path = workdir / f"{yaml_path.stem}.result.json"
        data = json.loads(result_path.read_text())
        e_hf.append(data["results"]["e_hf"])
        e_vqe.append(data["results"]["e_vqe"])
        print(f"  HF = {e_hf[-1]:.8f}  VQE = {e_vqe[-1]:.8f}")

    args.csv.write_text(
        "distance_A,e_hf_Ha,e_vqe_Ha\n"
        + "\n".join(f"{d:.6f},{eh:.8f},{ev:.8f}" for d, eh, ev in zip(distances, e_hf, e_vqe))
        + "\n"
    )
    print(f"Wrote {args.csv}")

    if args.no_plot:
        return 0

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("warning: matplotlib not installed; skipping plot.\n")
        return 0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(distances, e_hf, "o--", label="HF", color="tab:blue")
    ax.plot(distances, e_vqe, "s-", label="VQE (UCCSD)", color="tab:red")
    ax.set_xlabel("H–H distance / Å")
    ax.set_ylabel("Energy / Ha")
    ax.set_title("H2 / STO-3G dissociation curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=150)
    print(f"Wrote {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
