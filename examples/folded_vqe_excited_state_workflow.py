"""
Sample workflow for excited state calculation using Folded VQE (FS-VQE) as in https://pubs.acs.org/doi/10.1021/acs.jctc.3c01378
- First, run VQE to obtain ground state and energy E0
- Set delta = multiplier * |E0| (start with multiplier=0.01)
- Run FS-VQE for excited state, initializing with previous optimal parameters
- If converged folded energy < delta^2, accept; else, increase multiplier and repeat
"""

import numpy as np
from scipy.optimize import minimize
from qibochem.driver.hamiltonian import build_folded_hamiltonian
from qibochem.ansatz import hf_circuit, ucc_circuit
from qibochem.driver import Molecule
from qibochem.measurement import expectation

# --- Parameters for \omega search ---
DELTA_START_MULTIPLIER = 0.005
DELTA_STEP = 0.002
DELTA_MAX_MULTIPLIER = 0.20
MAX_ITER = 20

def get_ucc_excitations(n_qubits, n_electrons):
    # Copied from ucc_example1.py
    d_excitations = [
        (_i, _j, _a, _b)
        for _i in range(n_electrons)
        for _j in range(_i + 1, n_electrons)
        for _a in range(n_electrons, n_qubits)
        for _b in range(_a + 1, n_qubits)
        if (_i + _j + _a + _b) % 2 == 0 and ((_i % 2 + _j % 2) == (_a % 2 + _b % 2))
    ]
    s_excitations = [
        (_i, _a)
        for _i in range(n_electrons)
        for _a in range(n_electrons, n_qubits)
        if (_i + _a) % 2 == 0
    ]
    d_excitations = sorted(d_excitations, key=lambda x: (x[3] - x[2]) + (x[2] % 2))
    s_excitations = sorted(s_excitations, key=lambda x: (x[1] - x[0]) + (x[0] % 2))
    excitations = d_excitations + s_excitations
    n_unique_excitations = 5  # For LiH example
    return excitations, n_unique_excitations

def build_uccsd_circuit(n_qubits, n_electrons, excitations, parameters):
    circuit = hf_circuit(n_qubits, n_electrons)
    for _ex in excitations:
        circuit += ucc_circuit(n_qubits, _ex)
    coeff_dict = {1: (-1.0, 1.0), 2: (-0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25)}
    ucc_parameters = [
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[2],
        parameters[3],
        parameters[3],
        parameters[4],
        parameters[4],
    ]
    all_parameters = []
    for _ex, _parameter in zip(excitations, ucc_parameters):
        coeffs = coeff_dict[len(_ex) // 2]
        ucc_parameter = np.repeat(_parameter, len(coeffs))
        ucc_parameter *= coeffs
        all_parameters.append(ucc_parameter)
    all_parameters = [_x for _param in all_parameters for _x in _param]
    circuit.set_parameters(all_parameters)
    return circuit

def folded_vqe_excited_state_search_lih(verbose=True):
    # 1. Prepare molecule and Hamiltonian
    mol = Molecule(xyz_file="lih.xyz")
    mol.run_pyscf()
    mol.hf_embedding(active=[1, 2, 5])
    hamiltonian = mol.hamiltonian()
    n_qubits = mol.n_active_orbs
    n_electrons = mol.n_active_e
    excitations, n_unique_excitations = get_ucc_excitations(n_qubits, n_electrons)

    # 2. Ground state VQE
    def cost_gs(params):
        circuit = build_uccsd_circuit(n_qubits, n_electrons, excitations, params)
        return expectation(circuit, hamiltonian)
    params = np.random.rand(n_unique_excitations)
    res_gs = minimize(cost_gs, params)
    E0 = res_gs.fun
    gs_params = res_gs.x
    if verbose:
        print(f"Ground state energy: {E0:.8f}")

    # 3. Excited state search with delta sweep (FS-VQE)
    delta_multiplier = DELTA_START_MULTIPLIER
    params = gs_params.copy()
    abs_E0 = abs(E0)
    for i in range(MAX_ITER):
        delta = delta_multiplier * abs_E0
        omega = E0 + delta
        folded_ham = build_folded_hamiltonian(hamiltonian, omega)
        dense_ham = folded_ham.dense
        def cost_fs(params):
            circuit = build_uccsd_circuit(n_qubits, n_electrons, excitations, params)
            return expectation(circuit, dense_ham)

        res_fs = minimize(cost_fs, params)
        folded_energy = res_fs.fun
        if verbose:
            print(f"[FS-VQE] Iter {i+1}: delta_multiplier={delta_multiplier:.4f}, delta={delta:.6f}, folded_energy={folded_energy:.8f}")
        if folded_energy < delta**2:
            excited_energy = E0 + delta
            if verbose:
                print(f"Excited state found: E1 = {excited_energy:.8f}")
            return {
                "ground_energy": E0,
                "excited_energy": excited_energy,
                "delta": delta,
                "delta_multiplier": delta_multiplier,
                "folded_energy": folded_energy,
                "params": res_fs.x,
            }
        delta_multiplier += DELTA_STEP
        params = res_fs.x
        if delta_multiplier > DELTA_MAX_MULTIPLIER:
            break
    raise RuntimeError("Failed to find excited state: try increasing DELTA_MAX_MULTIPLIER or MAX_ITER.")

def main():
    result = folded_vqe_excited_state_search_lih()
    print("\nSummary:")
    print(f"Ground state energy: {result['ground_energy']:.8f}")
    print(f"Excited state energy: {result['excited_energy']:.8f}")
    print(f"delta: {result['delta']:.8f} (multiplier: {result['delta_multiplier']:.4f})")
    print(f"Folded energy: {result['folded_energy']:.8f}")

if __name__ == "__main__":
    main()