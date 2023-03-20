"""
Driver for obtaining molecular integrals from either PySCF or PSI4
"""

from pathlib import Path

import numpy as np
import openfermion

# import psi4
# import pyscf

import qibo
from qibo import hamiltonians
from qibo.symbols import X, Y, Z


class Molecule():
    """
    Class representing a single molecule
    """

    def __init__(self, geometry=None, charge=0, multiplicity=1, basis=None, xyz_file=None,
                 active=None):
        """
        Args:
            geometry: Molecular coordinates in OpenFermion format
                e.g. [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))]
            charge: Net charge of molecule
            multiplicity: Spin multiplicity of molecule
            basis: Atomic orbital basis set, used for the PySCF/PSI4 calculations
            xyz_file: .xyz file containing the molecular coordinates
                Comment line should follow "{charge} {multiplicity}"
            active: Iterable representing the set of MOs to be included in the quantum simulation

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
            assert Path(f"{xyz_file}").exists(), f"{xyz_file} not found!"
            self.process_xyz_file(xyz_file)
        if basis is None:
            # Default bais is STO-3G
            self.basis = 'sto-3g'
        else:
            self.basis = basis

        self.ca = None
        self.pa = None
        self.da = None
        self.nelec = None
        self.nalpha = None
        self.nbeta = None
        self.oei = None
        self.tei = None
        self.e_hf = None
        self.e_nuc = None
        self.norb = None
        self.nso = None
        self.overlap = None
        self.eps = None
        self.fa = None
        self.hcore = None
        self.ja = None
        self.ka = None
        self.aoeri = None

        # For HF embedding
        self.active = active # List of active MOs included in the active space
        self.frozen = None

        self.inactive_energy = None
        self.embed_oei = None
        self.embed_tei = None

        self.n_active_e = None
        self.n_active_orbs = None # Number of spin-orbitals in the active space 


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
        spin = self.multiplicity - 1 # PySCF spin is 2S
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
        self.ca = np.asarray(pyscf_job.mo_coeff) # MO coeffcients
        self.nelec = pyscf_mol.nelec
        self.nalpha = self.nelec[0]
        self.nbeta = self.nelec[1]
        self.e_hf = pyscf_job.e_tot # HF energy
        self.e_nuc = pyscf_mol.energy_nuc()
        self.norb = self.ca.shape[1]
        self.nso = 2*self.norb
        self.overlap = np.asarray(pyscf_mol.intor('int1e_ovlp'))
        self.eps = np.asarray(pyscf_job.mo_energy)
        self.fa = pyscf_job.get_fock()
        self.hcore = pyscf_job.get_hcore()
        self.ja = pyscf_job.get_j()
        self.ka = pyscf_job.get_k()
        self.aoeri = np.asarray(pyscf_mol.intor('int2e'))

        ca_occ = self.ca[:, 0:self.nalpha]
        self.pa = ca_occ @ ca_occ.T
        self.da = self.ca.T @ self.overlap @ self.pa @ self.overlap @ self.ca

        # 1-electron integrals
        oei = np.asarray(pyscf_mol.intor('int1e_kin')) + np.asarray(pyscf_mol.intor('int1e_nuc'))
        #oei = np.asarray(pyscf_mol.get_hcore())
        oei = np.einsum("ab,bc->ac", oei, self.ca)
        oei = np.einsum("ab,ac->bc", self.ca, oei)
        self.oei = oei

        # Two electron integrals
        #tei = np.asarray(pyscf_mol.intor('int2e'))
        eri = pyscf.ao2mo.kernel(pyscf_mol, self.ca)
        eri4 = pyscf.ao2mo.restore('s1', eri, self.norb)
        tei = np.einsum("pqrs->prsq", eri4)
        self.tei = tei


    def run_psi4(self, output=None):
        """
        Run a Hartree-Fock calculation with PSI4 to obtain the molecular quantities and
            molecular integrals

        Args:
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
        opts = {'basis': self.basis, 'scf_type': 'direct', 'reference': 'rhf', 'save_jk': True}
        psi4.core.clean()
        psi4.set_memory('500 MB')
        psi4.set_options(opts)
        # Keep the output file of the PSI4 calculation?
        if output is None:
            # Doesn't work on Windows!
            # See: https://psicode.org/psi4manual/master/api/psi4.core.be_quiet.html
            try:
                psi4.core.be_quiet()
            except RuntimeError:
                psi4.core.set_output_file('psi4_output.dat', False)
        else:
            psi4.core.set_output_file(output, False)

        # Run HF calculation with PSI4
        psi4_mol = psi4.geometry(mol_string)
        ehf, wavefn = psi4.energy('hf', return_wfn=True)

        # Save 1- and 2-body integrals
        ca = wavefn.Ca() # MO coefficients
        self.ca = np.asarray(ca)
        # 1- electron integrals
        oei = wavefn.H() # 'Core' (potential + kinetic) integrals
        # Convert from AO->MO basis
        oei = np.einsum("ab,bc->ac", oei, self.ca)
        oei = np.einsum("ab,ac->bc", self.ca, oei)

        # 2- electron integrals
        mints = psi4.core.MintsHelper(wavefn.basisset())
        tei = np.asarray(mints.mo_eri(ca, ca, ca, ca)) # Need original C_a array, not a np.array
        tei = np.einsum("pqrs->prsq", tei)

        # Fill in the class attributes
        self.nelec = (wavefn.nalpha(), wavefn.nbeta())
        self.nalpha = self.nelec[0]
        self.nbeta = self.nelec[1]
        self.e_hf = ehf
        self.e_nuc = psi4_mol.nuclear_repulsion_energy()
        self.norb = wavefn.nmo()
        self.nso = 2*self.norb
        self.oei = oei
        self.tei = tei
        self.aoeri = np.asarray(mints.ao_eri())
        self.overlap = np.asarray(wavefn.S())
        self.eps = np.asarray(wavefn.epsilon_a())
        self.fa = np.asarray(wavefn.Fa())
        self.hcore = np.asarray(wavefn.H())

        self.ja = np.asarray(wavefn.jk().J()[0])
        self.ka = np.asarray(wavefn.jk().K()[0])

        ca_occ = self.ca[:, 0:self.nalpha]
        self.pa = ca_occ @ ca_occ.T
        self.da = self.ca.T @ self.overlap @ self.pa @ self.overlap @ self.ca


    # HF embedding functions
    def inactive_fock_matrix(self, frozen):
        """
        Returns the full inactive Fock matrix

        Args:
            frozen: Iterable representing the occupied orbitals to be removed from the simulation
        """
        # Copy the original OEI as a starting point
        inactive_fock = np.copy(self.oei)

        # Add the two-electron operator, summed over the frozen orbitals
        # Iterate over the active orbitals
        for _p in range(self.norb):
            for _q in range(self.norb):
                # Iterate over the inactive orbitals
                for _orb in frozen:
                    # Add (2J - K) using TEI (in OpenFermion format)
                    inactive_fock[_p][_q] += (2*self.tei[_orb][_p][_q][_orb]
                                              - self.tei[_orb][_p][_orb][_q])
        return inactive_fock


    def hf_embedding(self, active=None, frozen=None):
        """
        Turns on HF embedding for a given active/frozen space, i.e.
        fills the class attributes: inactive_energy, embed_oei, and embed_tei

        Args:
            active: Iterable representing the active-space for quantum simulation
            frozen: Iterable representing the occupied orbitals to be removed from the simulation
        """
        # Default arguments for active and frozen
        if active is None:
            if self.active is None:
                active = list(range(self.norb))
            else:
                active = self.active
        if frozen is None:
            if self.frozen is None:
                frozen = [_i for _i in range(self.nalpha) if _i not in active]
            else:
                frozen = self.frozen

        # Check that arguments are valid
        assert max(active) < self.norb and min(active) >= 0, ("Active space must be between 0 "
                                                              "and the number of MOs")
        if frozen:
            assert not(set(active) & set(frozen)), "Active and frozen space cannot overlap"
            assert max(frozen)+1 < sum(self.nelec)//2 and min(frozen) >= 0, ("Frozen orbitals must be "
                                                                             "occupied orbitals")

        # Build the inactive Fock matrix first
        inactive_fock = self.inactive_fock_matrix(frozen)

        # Calculate the inactive Fock energy
        # Only want frozen part of original OEI and inactive Fock matrix
        _oei = self.oei[np.ix_(frozen, frozen)]
        _inactive_fock = inactive_fock[np.ix_(frozen, frozen)]
        self.inactive_energy = np.einsum('ii->', _oei + _inactive_fock)

        # Keep only the active part
        self.embed_oei = inactive_fock[np.ix_(active, active)]
        self.embed_tei = self.tei[np.ix_(active, active, active, active)]

        # Update class attributes
        self.active = active
        self.frozen = frozen
        self.n_active_orbs = 2*len(active)
        self.n_active_e = sum(self.nelec) - 2*len(self.frozen)


    def fermionic_hamiltonian(
        self,
        oei=None,
        tei=None,
        constant=None,
    ) -> openfermion.InteractionOperator:
        """
        Molecular Hamiltonian using the OEI/TEI given in MO basis

        Args:
            oei: 1-electron integrals in OpenFermion notation. Default: self.oei
            tei: 2-electron integrals in OpenFermion notation. Default: self.tei
            constant: For inactive Fock energy if embedding used. Default: 0.0

        Returns:
            hamiltonian (InteractionOperator): Molecular Hamiltonian
        """
        # Define default variables
        if oei is None:
            oei = self.oei
        if tei is None:
            tei = self.tei
        if constant is None:
            constant = 0.0

        # Convert the 1-/2- electron integrals from MO basis to SO basis
        oei_so, tei_so = openfermion.ops.representations.get_tensors_from_integrals(oei, tei)
        # tei_so already multiplied by 0.5, no need to include in InteractionOperator
        return openfermion.InteractionOperator(self.e_nuc+constant, oei_so, tei_so)


    @staticmethod
    def qubit_hamiltonian(
        fermion_hamiltonian,
        ferm_qubit_map: str = "jw"
    ) -> openfermion.QubitOperator:
        """
        Converts the molecular Hamiltonian to a QubitOperator

        Args:
            fermion_hamiltonian: Molecular Hamiltonian as a InteractionOperator/FermionOperator
            ferm_qubit_map: Which Fermion->Qubit mapping to use

        Returns:
            qubit_operator : Molecular Hamiltonian as a QubitOperator
        """
        # Map the fermionic molecular Hamiltonian to a QubitHamiltonian
        if ferm_qubit_map == "jw":
            q_hamiltonian = openfermion.jordan_wigner(fermion_hamiltonian)
        elif ferm_qubit_map == "bk": # Not tested!
            q_hamiltonian = openfermion.bravyi_kitaev(fermion_hamiltonian)
        else:
            raise NameError("Unknown fermion->qubit mapping!")

        return q_hamiltonian


    @staticmethod
    def symbolic_hamiltonian(
        q_hamiltonian: openfermion.QubitOperator
    ) -> qibo.hamiltonians.SymbolicHamiltonian:
        """
        Returns the molecular Hamiltonian as a Qibo SymbolicHamiltonian object

        Args:
            q_hamiltonian: Molecular Hamiltonian given as a QubitOperator

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

        # Sums over each individual Pauli string in the QubitOperator
        symbolic_ham = sum(parse_pauli_string(pauli_string, coeff)
                           # Iterate over all operators
                           for operator in q_hamiltonian.get_operators()
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

        return hamiltonian.expectation(state_ket)


    def eigenvalues(self, hamiltonian=None):
        """
        Exact eigenvalues of a molecular Hamiltonian


        Args:
            hamiltonian (openfermion.QubitOperator): Defaults to the Molecular Hamiltonian
        """
        if hamiltonian is None:
            hamiltonian = self.fermionic_hamiltonian()

        qubit_ham = self.qubit_hamiltonian(hamiltonian)
        sym_hamiltonian = self.symbolic_hamiltonian(qubit_ham)

        return sym_hamiltonian.eigenvalues()
