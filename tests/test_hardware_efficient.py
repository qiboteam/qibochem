import numpy as np
from qibo import gates, models
from scipy.optimize import minimize

from qibochem.ansatz import hardware_efficient
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation


def test_hea_ansatz():
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74804))])
    mol.run_pyscf()
    mol_classical_hf_energy = mol.e_hf
    mol_sym_ham = mol.hamiltonian("s")

    nlayers = 1
    nqubits = mol.nso
    ntheta = 2 * nqubits * nlayers
    theta = np.zeros(ntheta)

    hea_ansatz = hardware_efficient.hea(nlayers, nqubits)
    qc = models.Circuit(nqubits)
    qc.add(gates.X(_i) for _i in range(sum(mol.nelec)))
    qc.add(hea_ansatz)
    qc.set_parameters(theta)

    hf_energy = expectation(qc, mol_sym_ham)
    assert mol_classical_hf_energy == pytest.approx(hf_energy)


def test_vqe_hea_ansatz():
    def test_vqe_hea_ansatz_cost(parameters, circuit, hamiltonian):
        circuit.set_parameters(parameters)
        return expectation(circuit, hamiltonian)

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.75))])
    mol.run_pyscf()
    mol_classical_hf_energy = mol.e_hf
    mol_sym_ham = mol.hamiltonian("s")

    nlayers = 2
    nqubits = mol.nso
    ntheta = 2 * nqubits * nlayers
    theta = np.full(ntheta, np.pi / 4)

    hea_ansatz = hardware_efficient.hea(nlayers, nqubits)
    qc = models.Circuit(nqubits)
    qc.add(gates.X(_i) for _i in range(sum(mol.nelec)))
    qc.add(hea_ansatz)
    qc.set_parameters(theta)

    vqe_object = minimize(test_vqe_hea_ansatz_cost, theta, args=(qc, mol_sym_ham), method="Powell")

    vqe_hf_energy = vqe_object.fun
    assert mol_classical_hf_energy == pytest.approx(vqe_hf_energy)
