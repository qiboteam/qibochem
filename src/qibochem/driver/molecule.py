"""
Driver for obtaining molecular integrals from either PySCF or PSI4
"""

from pathlib import Path

import numpy as np
import openfermion as of

# import psi4
# import pyscf

import qibo
from qibo import hamiltonians
from qibo.symbols import X, Y, Z


class Molecule():
    """
    Class representing a single molecule
    """

    def __init__(self, geometry=None, charge=0, multiplicity=1, basis=None, xyz_file=None):
        """
        Args:
            geometry: Molecular coordinates in OpenFermion format
                e.g. [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))]
            charge: Net charge of molecule
            multiplicity: Spin multiplicity of molecule
            basis: Atomic orbital basis set, used for the PySCF/PSI4 calculations
            xyz_file: .xyz file containing the molecular coordinates
                Comment line should follow "{charge} {multiplicity}"

        Example:
            .. testcode::
                TODO
        """
        # Basic properties
        # Define using the function arguments if xyz_file not given
        if xyz_file is None:
            self.geometry = geometry
            self.charge = charge
            self.multiplicity = multiplicity
        else:
            # Check if xyz_file exists, then fill in the Molecule attributes
            assert Path(f"./{xyz_file}").exists(), f"{xyz_file} not found!"
            self.process_xyz_file(xyz_file)
        # if rhf is None:
        #     rhf = (multiplicity == 1) # bool(multiplicity == 1)
        # self.rhf = rhf # Reference wave function is restricted Hartree-Fock
        if basis is None:
            # Default bais is STO-3G
            self.basis = 'sto-3g'

        self.ca = None
        self.cb = None
        self.pa = None
        self.pb = None
        self.da = None
        self.db = None
        self.nelec = None
        self.nalpha = None
        self.nbeta = None
        self.oei = None
        self.tei = None
        self.e_hf = None
        self.e_nuc = None
        self.nbf = None
        self.norb = None
        self.nso = None
        self.overlap = None
        self.eps = None
        self.fa = None
        self.fb = None
        self.hcore = None
        self.ja = None
        self.jb = None
        self.ka = None
        self.kb = None
        self.aoeri = None


    def process_xyz_file(self, xyz_file):
        """
        Reads a .xyz file to obtain and set the molecular coordinates (in OpenFermion format),
            charge, and multiplicity

        Args:
            xyz_file: .xyz file for molecule. Comment line should follow "{charge} {multiplicity}"
        """
        with open(xyz_file, "r", encoding='utf-8') as file_handler:
            # First two lines: # atoms and comment line (charge, multiplicity)
            _n_atoms = int(file_handler.readline()) # Not needed/used
            _charge, _multiplicity = [int(_num) for _num in file_handler.readline().split()]

            # Start reading xyz coordinates from the 3rd line onwards
            _geometry = []
            for line in file_handler:
                split_line = line.split()
                # OpenFermion format: [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7)), ...]
                atom_xyz = [split_line[0], tuple(float(_xyz) for _xyz in split_line[1:4])]
                _geometry.append(tuple(atom_xyz))

        # Set the class attributes
        self.charge = _charge
        self.multiplicity = _multiplicity
        self.geometry = _geometry


    def run_pyscf(self, max_scf_cycles=50):
        """
        Run a Hartree-Fock calculation with PySCF to obtain molecule quantities and
            molecular integrals

        Args:
            basis: Atomic orbital basis set
        """
        import pyscf

        # Set up and run PySCF calculation
        geom_string = "".join("{} {:.6f} {:.6f} {:.6f} ; ".format(_atom[0], *_atom[1])
                              for _atom in self.geometry)
        spin = int((self.multiplicity - 1) // 2)
        pyscf_mol = pyscf.gto.M(charge=self.charge,
                                spin=spin,
                                atom=geom_string,
                                basis=self.basis,
                                symmetry='C1')
        
        pyscf_job = pyscf.scf.RHF(pyscf_mol)
        pyscf_job.max_cycle = max_scf_cycles
        pyscf_job.run()
        #print(dir(pyscf_job))

        # Save results from HF calculation
        ehf = pyscf_job.e_tot # HF energy

        ca = np.asarray(pyscf_job.mo_coeff) # MO coeffcients

        # 1-electron integrals
        oei = np.asarray(pyscf_mol.intor('int1e_kin')) + np.asarray(pyscf_mol.intor('int1e_nuc'))
        #oei = np.asarray(pyscf_mol.get_hcore())
        oei = np.einsum("ab,bc->ac", oei, ca)
        oei = np.einsum("ab,ac->bc", ca, oei)

        # Two electron integrals
        #tei = np.asarray(pyscf_mol.intor('int2e'))
        eri = pyscf.ao2mo.kernel(pyscf_mol, ca)
        eri4 = pyscf.ao2mo.restore('s1', eri, ca.shape[1])
        tei = np.einsum("pqrs->prsq", eri4)

        # Fill in the class attributes
        self.ca = ca
        self.nelec = pyscf_mol.nelec
        self.nalpha = self.nelec[0]
        self.nbeta = self.nelec[1]
        self.e_hf = ehf
        self.e_nuc = pyscf_mol.energy_nuc()
        self.nbf = ca.shape[0]
        self.norb = ca.shape[1]
        self.nso = 2*ca.shape[1]
        self.oei = oei
        self.tei = tei
        self.overlap = np.asarray(pyscf_mol.intor('int1e_ovlp'))
        self.eps = np.asarray(pyscf_job.mo_energy)
        self.fa = pyscf_job.get_fock()
        self.hcore = pyscf_job.get_hcore()
        self.ja = pyscf_job.get_j()
        self.ka = pyscf_job.get_k()
        self.aoeri = np.asarray(pyscf_mol.intor('int2e'))

        ca_occ = ca[:, 0:self.nalpha]
        pa = ca_occ @ ca_occ.T
        self.pa = pa
        da = ca.T @ self.overlap @ pa @ self.overlap @ ca
        self.da = da
        if self.multiplicity == 1:
            self.cb = ca
            self.db = da
            #self.fb = fa


    def run_psi4(self, basis='sto-3g', output=None):# 'psi4_output.out'):
        """
        Run a Hartree-Fock calculation with PSI4 to obtain the molecular quantities and
            molecular integrals

        Args:
            basis: Atomic orbital basis set
            output: Name of PSI4 output file. None suppresses the output on non-Windows systems,
                and uses 'psi4_output.dat' otherwise
        """
        import psi4

        # PSI4 input string
        chgmul_string = f"{self.charge} {self.multiplicity} \n"
        geom_string = "\n".join("{} {:.6f} {:.6f} {:.6f}".format(_atom[0], *_atom[1])
                                for _atom in self.geometry)
        opt1_string = "\n\nunits angstrom\nsymmetry c1\n"
        mol_string = f"{chgmul_string}{geom_string}{opt1_string}"
        # PSI4 calculation options
        opts = {'basis': basis, 'scf_type': 'direct'} #, 'reference': 'rhf'} # Default: RHF?
        psi4.core.clean()
        psi4.set_memory('500 MB')
        psi4.set_options(opts)
        # Keep the output file of the PSI4 calculation?
        if output is None:
            # Doesn't work on Windows!
            # See: https://psicode.org/psi4manual/master/api/psi4.core.be_quiet.html
            try:
                psi4.core.be_quiet()
            except:
                psi4.core.set_output_file('psi4_output.dat', False)
        else:
            psi4.core.set_output_file(output, False)

        # Run HF calculation with PSI4
        psi4_mol = psi4.geometry(mol_string)
        ehf, wavefn = psi4.energy('hf', return_wfn=True)

        # Save 1- and 2-body integrals
        # 1- electron integrals
        ca = wavefn.Ca() # MO coefficients
        oei = wavefn.H() # 'Core' (potential + kinetic) integrals
        # Convert from AO->MO basis
        oei = np.einsum("ab,bc->ac", oei, np.asarray(ca))
        oei = np.einsum("ab,ac->bc", np.asarray(ca), oei)
        #print(oei)

        # 2- electron integrals
        mints = psi4.core.MintsHelper(wavefn.basisset())
        tei = np.asarray(mints.mo_eri(ca, ca, ca, ca))
        tei = np.einsum("pqrs->prsq", tei)
        #print(tei)

        # Fill in the class attributes
        self.ca = np.asarray(ca)
        self.nelec = (wavefn.nalpha(), wavefn.nbeta())
        self.nalpha = self.nelec[0]
        self.nbeta = self.nelec[1]
        self.e_hf = ehf
        self.e_nuc = psi4_mol.nuclear_repulsion_energy()
        self.nbf = ca.shape[0]
        self.norb = wavefn.nmo()
        self.nso = 2*wavefn.nmo()
        self.oei = oei
        self.tei = tei
        self.aoeri = np.asarray(mints.ao_eri())
        self.overlap = np.asarray(wavefn.S())
        self.eps = np.asarray(wavefn.epsilon_a())
        self.fa = np.asarray(wavefn.Fa())
        self.hcore = np.asarray(wavefn.H())

        da = np.asarray(wavefn.Da())
        # db = np.asarray(wavefn.Db())
        ja = np.einsum('pqrs,rs->pq', self.aoeri, da, optimize=True)
        ka = np.einsum('prqs,rs->pq', self.aoeri, da, optimize=True)
        self.ja = ja
        self.ka = ka

        ca_occ = self.ca[:, 0:self.nalpha]
        pa = ca_occ @ ca_occ.T
        self.pa = pa
        da = self.ca.T @ self.overlap @ pa @ self.overlap @ self.ca
        self.da = da
        if self.multiplicity == 1:
            self.cb = ca
            self.db = da


    def get_ofdata(self):
        """
        Creates an OpenFermion MolecularData object using PySCF or PSI4 data
        """

        if self.e_hf is None:
            raise Exception('Run PySCF or PSI4 first to obtain molecular data')

        # Initialize OpenFermion MolecularData object
        molecule_data = of.chem.MolecularData(self.geometry,
                                              basis=self.basis,
                                              multiplicity=self.multiplicity,
                                              charge=self.charge)

        molecule_data.one_body_integrals = self.oei
        molecule_data.two_body_integrals = self.tei

        molecule_data.hf_energy = self.e_hf
        molecule_data.nuclear_repulsion = self.e_nuc
        molecule_data.n_orbitals = self.norb
        molecule_data.n_qubits = 2 * molecule_data.n_orbitals

        molecule_data.overlap_integrals = self.overlap
        molecule_data.orbital_energies = self.eps

        return molecule_data


    # Functions to obtain the molecular Hamiltonian and convert it to a Qibo SymbolicHamiltonian
    def qubit_molecular_hamiltonian(self, ferm_qubit_map: str) -> of.QubitOperator:
        """
        Converts the molecular Hamiltonian to a OpenFermion QubitOperator

        Args:
            ferm_qubit_map (str): Which Fermion->Qubit mapping to use

        Returns:
            qubit_operator (of.QubitOperator): Molecular Hamiltonian as a QubitOperator
        """
        # Build the OpenFermion MolecularData class and get the Hamiltonian
        molecular_data = self.get_ofdata()
        fermion_hamiltonian = molecular_data.get_molecular_hamiltonian()

        # Map the fermionic molecular Hamiltonian to a QubitHamiltonian
        if ferm_qubit_map == "jw":
            qubit_hamiltonian = of.jordan_wigner(fermion_hamiltonian)
        elif ferm_qubit_map == "bk": # Not tested!
            qubit_hamiltonian = of.bravyi_kitaev(fermion_hamiltonian)
        else:
            raise Exception("Unknown fermion->qubit mapping!")

        return qubit_hamiltonian


    def symbolic_hamiltonian(
        self,
        ferm_qubit_map: str="jw"
    ) -> qibo.hamiltonians.SymbolicHamiltonian:
        """
        Returns the molecular Hamiltonian as a Qibo SymbolicHamiltonian object

        Args:
            ferm_qubit_map (str): Which Fermion->Qubit mapping to use

        Returns:
            qibo.hamiltonians.SymbolicHamiltonian
        """
        def parse_pauli_string(
            pauli_string: 'tuple of tuples',
            coeff: float
        ) -> qibo.symbols.Symbol:
            """
            Helper function: Converts a single Pauli string to a Qibo Symbol

            Args:
                pauli_string (tuple of tuples): Indicate what gates to apply onto which qubit
                    e.g. ((0, 'Z'), (2, 'Z'))
                coeff (float): Coefficient of the Pauli string

            Returns:
                qibo.symbols.Symbol for a single Pauli string, e.g. -0.04*X0*X1*Y2*Y3
            """
            # Dictionary for converting
            xyz_to_symbol = {'X': X, 'Y': Y, 'Z': Z}
            # Check that pauli_string is non-empty
            if pauli_string:
                # pauli_string format: ((0, 'Y'), (1, 'Y'), (3, 'X'))
                qibo_pauli_string = 1.0
                for p_letter in pauli_string:
                    qibo_pauli_string *= xyz_to_symbol[p_letter[1]](p_letter[0])
                # Include coefficient after all gates
                qibo_pauli_string = coeff * qibo_pauli_string
            else:
                # Empty word, i.e. constant term in Hamiltonian
                qibo_pauli_string = coeff
            return qibo_pauli_string

        # Molecular Hamiltonian as a QubitOperator
        qubit_hamiltonian = self.qubit_molecular_hamiltonian(ferm_qubit_map)

        # Sums over each individual Pauli string in the QubitOperator
        symbolic_ham = sum(parse_pauli_string(pauli_string, coeff)
                           # Iterate over all operators
                           for operator in qubit_hamiltonian.get_operators()
                           # .terms gives one operator as a dictionary with one entry
                           for pauli_string, coeff in operator.terms.items()
                           )

        # Map the QubitHamiltonian to a Qibo SymbolicHamiltonian and return it
        return hamiltonians.SymbolicHamiltonian(symbolic_ham)


    def expectation_value(
            self,
            circuit: qibo.models.Circuit,
            hamiltonian: qibo.hamiltonians.SymbolicHamiltonian
        ) -> float:
        """
        Calculate expectation value of Hamiltonian using the state vector from running a circuit

        Args:
            circuit (qibo.models.Circuit): Quantum circuit ansatz
            hamiltonian (qibo.hamiltonians.SymbolicHamiltonian): Molecular Hamiltonian

        Returns:
            Hamiltonian expectation value (float)
        """
        circuit_result = circuit(nshots=1)
        state_ket = circuit_result.state()

        # Qibo 0.1.8/0.1.9 bug: Dense matrix formed from SymbolicHamiltonian when calculating
        #    expectation values, so need to do it manually instead
        if qibo.__version__ in ('0.1.8', '0.1.9'):
            state_bra = np.conj(state_ket)
            return np.real(np.sum(state_bra * (hamiltonian @ state_ket)))
        # Otherwise, the expectation method of a Hamiltonian can be used directly
        return hamiltonian.expectation(state_ket)


    def exact_eigenvalues(self, hamiltonian: of.QubitOperator):
        """
        Exact eigenvalues of a QubitOperator Hamiltonian
        Note: Probably only needed for Qibo 0.1.8?

        Args:
            hamiltonian (of.QubitOperator): Molecular Hamiltonian
        """
        # Bug in Qibo 0.1.8 and 0.1.9: See comment in expectation_value function
        if qibo.__version__ in ('0.1.8', '0.1.9'):
            # Use the scipy sparse matrix library
            from scipy.sparse import linalg

            hamiltonian_matrix = of.get_sparse_operator(hamiltonian)
            eigenvalues, _ = linalg.eigsh(hamiltonian_matrix, k=6, which="SA")
        # Otherwise, the expectation method of a Hamiltonian can be used directly
        else:
            sym_hamiltonian = self.symbolic_hamiltonian()
            eigenvalues = sym_hamiltonian.eigenvalues()

        return eigenvalues
