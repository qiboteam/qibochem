"""
Driver for obtaining molecular integrals from either PySCF or PSI4
"""

from pathlib import Path

import numpy as np
import openfermion
import qibo
from qibo.hamiltonians import SymbolicHamiltonian

from qibochem.driver.hamiltonian import (
    fermionic_hamiltonian,
    qubit_hamiltonian,
    symbolic_hamiltonian,
)


class Molecule:
    """
    Class representing a single molecule

    Args:
        geometry (list): Molecular coordinates in OpenFermion format,  e.g. ``[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))]``
        charge (int): Net electronic charge of molecule
        multiplicity (int): Spin multiplicity of molecule, given as 2S + 1, where S is half the number of unpaired electrons
        basis (str): Atomic orbital basis set, used for the PySCF/PSI4 calculations. Default: "STO-3G" (minimal basis)
        xyz_file (str): .xyz file containing the molecular coordinates. The comment line can be used to define the electronic
            charge and spin multiplity if it is given in this format: "{charge} {multiplicity}"
        active: Iterable representing the set of MOs to be included in the quantum simulation
            e.g. ``list(range(3,6))`` for an active space with orbitals 3, 4 and 5.

    """

    def __init__(self, geometry=None, charge=0, multiplicity=1, basis=None, xyz_file=None, active=None):
        # Basic properties
        # Define using the function arguments if xyz_file not given
        if xyz_file is None:
            self.geometry = geometry
            self.charge = charge
            self.multiplicity = multiplicity
        else:
            # Check if xyz_file exists, then fill in the Molecule attributes
            assert Path(f"{xyz_file}").exists(), f"{xyz_file} not found!"
            self._process_xyz_file(xyz_file, charge, multiplicity)
        if basis is None:
            # Default bais is STO-3G
            self.basis = "sto-3g"
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
        self.active = active  # List of active MOs included in the active space
        self.frozen = None

        self.inactive_energy = None
        self.embed_oei = None
        self.embed_tei = None

        self.n_active_e = None
        self.n_active_orbs = None  # Number of spin-orbitals in the active space

    def _process_xyz_file(self, xyz_file, charge, multiplicity):
        """
        Reads a .xyz file to obtain and set the molecular coordinates (in OpenFermion format),
            charge, and multiplicity

        Args:
            xyz_file: .xyz file for molecule. Comment line should follow "{charge} {multiplicity}"
        """
        with open(xyz_file, encoding="utf-8") as file_handler:
            # First two lines: # atoms and comment line (charge, multiplicity)
            _n_atoms = int(file_handler.readline())  # Not needed/used

            # Try to read charge and multiplicity from comment line
            split_line = [int(_num) for _num in file_handler.readline().split()]
            if len(split_line) == 2:
                # Format of comment line matches (charge, multiplicity):
                _charge, _multiplicity = split_line
            else:
                # Otherwise, use the default (from __init__) values of 0 and 1
                _charge, _multiplicity = charge, multiplicity

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
            max_scf_cycles: Maximum number of SCF cycles in PySCF
        """
        import pyscf

        # Set up and run PySCF calculation
        geom_string = "".join("{} {:.6f} {:.6f} {:.6f} ; ".format(_atom[0], *_atom[1]) for _atom in self.geometry)
        spin = self.multiplicity - 1  # PySCF spin is 2S
        pyscf_mol = pyscf.gto.M(charge=self.charge, spin=spin, atom=geom_string, basis=self.basis, symmetry="C1")

        pyscf_job = pyscf.scf.RHF(pyscf_mol)
        pyscf_job.max_cycle = max_scf_cycles
        pyscf_job.run()
        # print(dir(pyscf_job))

        # Save results from HF calculation
        self.ca = np.asarray(pyscf_job.mo_coeff)  # MO coeffcients
        self.nelec = pyscf_mol.nelec
        self.nalpha = self.nelec[0]
        self.nbeta = self.nelec[1]
        self.e_hf = pyscf_job.e_tot  # HF energy
        self.e_nuc = pyscf_mol.energy_nuc()
        self.norb = self.ca.shape[1]
        self.nso = 2 * self.norb
        self.overlap = np.asarray(pyscf_mol.intor("int1e_ovlp"))
        self.eps = np.asarray(pyscf_job.mo_energy)
        self.fa = pyscf_job.get_fock()
        self.hcore = pyscf_job.get_hcore()
        self.ja = pyscf_job.get_j()
        self.ka = pyscf_job.get_k()
        self.aoeri = np.asarray(pyscf_mol.intor("int2e"))

        ca_occ = self.ca[:, 0 : self.nalpha]
        self.pa = ca_occ @ ca_occ.T
        self.da = self.ca.T @ self.overlap @ self.pa @ self.overlap @ self.ca

        # 1-electron integrals
        oei = np.asarray(pyscf_mol.intor("int1e_kin")) + np.asarray(pyscf_mol.intor("int1e_nuc"))
        # oei = np.asarray(pyscf_mol.get_hcore())
        oei = np.einsum("ab,bc->ac", oei, self.ca)
        oei = np.einsum("ab,ac->bc", self.ca, oei)
        self.oei = oei

        # Two electron integrals
        # tei = np.asarray(pyscf_mol.intor('int2e'))
        eri = pyscf.ao2mo.kernel(pyscf_mol, self.ca)
        eri4 = pyscf.ao2mo.restore("s1", eri, self.norb)
        tei = np.einsum("pqrs->prsq", eri4)
        self.tei = tei

    # def run_psi4(self, output=None):
    #     """
    #     Run a Hartree-Fock calculation with PSI4 to obtain the molecular quantities and
    #         molecular integrals

    #     Args:
    #         output: Name of PSI4 output file. ``None`` suppresses the output on non-Windows systems,
    #             and uses ``psi4_output.dat`` otherwise
    #     """
    #     import psi4  # pylint: disable=import-error

    #     # PSI4 input string
    #     chgmul_string = f"{self.charge} {self.multiplicity} \n"
    #     geom_string = "\n".join("{} {:.6f} {:.6f} {:.6f}".format(_atom[0], *_atom[1]) for _atom in self.geometry)
    #     opt1_string = "\n\nunits angstrom\nsymmetry c1\n"
    #     mol_string = f"{chgmul_string}{geom_string}{opt1_string}"
    #     # PSI4 calculation options
    #     opts = {"basis": self.basis, "scf_type": "direct", "reference": "rhf", "save_jk": True}
    #     psi4.core.clean()
    #     psi4.set_memory("500 MB")
    #     psi4.set_options(opts)
    #     # Keep the output file of the PSI4 calculation?
    #     if output is None:
    #         # Doesn't work on Windows!
    #         # See: https://psicode.org/psi4manual/master/api/psi4.core.be_quiet.html
    #         try:
    #             psi4.core.be_quiet()
    #         except RuntimeError:
    #             psi4.core.set_output_file("psi4_output.dat", False)
    #     else:
    #         psi4.core.set_output_file(output, False)

    #     # Run HF calculation with PSI4
    #     psi4_mol = psi4.geometry(mol_string)
    #     ehf, wavefn = psi4.energy("hf", return_wfn=True)

    #     # Save 1- and 2-body integrals
    #     ca = wavefn.Ca()  # MO coefficients
    #     self.ca = np.asarray(ca)
    #     # 1- electron integrals
    #     oei = wavefn.H()  # 'Core' (potential + kinetic) integrals
    #     # Convert from AO->MO basis
    #     oei = np.einsum("ab,bc->ac", oei, self.ca)
    #     oei = np.einsum("ab,ac->bc", self.ca, oei)

    #     # 2- electron integrals
    #     mints = psi4.core.MintsHelper(wavefn.basisset())
    #     tei = np.asarray(mints.mo_eri(ca, ca, ca, ca))  # Need original C_a array, not a np.array
    #     tei = np.einsum("pqrs->prsq", tei)

    #     # Fill in the class attributes
    #     self.nelec = (wavefn.nalpha(), wavefn.nbeta())
    #     self.nalpha = self.nelec[0]
    #     self.nbeta = self.nelec[1]
    #     self.e_hf = ehf
    #     self.e_nuc = psi4_mol.nuclear_repulsion_energy()
    #     self.norb = wavefn.nmo()
    #     self.nso = 2 * self.norb
    #     self.oei = oei
    #     self.tei = tei
    #     self.aoeri = np.asarray(mints.ao_eri())
    #     self.overlap = np.asarray(wavefn.S())
    #     self.eps = np.asarray(wavefn.epsilon_a())
    #     self.fa = np.asarray(wavefn.Fa())
    #     self.hcore = np.asarray(wavefn.H())

    #     self.ja = np.asarray(wavefn.jk().J()[0])
    #     self.ka = np.asarray(wavefn.jk().K()[0])

    #     ca_occ = self.ca[:, 0 : self.nalpha]
    #     self.pa = ca_occ @ ca_occ.T
    #     self.da = self.ca.T @ self.overlap @ self.pa @ self.overlap @ self.ca

    # HF embedding functions
    def _inactive_fock_matrix(self, frozen):
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
                    inactive_fock[_p][_q] += 2 * self.tei[_orb][_p][_q][_orb] - self.tei[_orb][_p][_orb][_q]
        return inactive_fock

    def hf_embedding(self, active=None, frozen=None):
        """
        Turns on HF embedding for a given active/frozen space, and fills in the class attributes:
            ``inactive_energy``, ``embed_oei``, and ``embed_tei``.

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
        assert max(active) < self.norb and min(active) >= 0, "Active space must be between 0 " "and the number of MOs"
        if frozen:
            assert not (set(active) & set(frozen)), "Active and frozen space cannot overlap"
            assert max(frozen) + 1 < sum(self.nelec) // 2 and min(frozen) >= 0, (
                "Frozen orbitals must" " be occupied orbitals"
            )

        # Build the inactive Fock matrix first
        inactive_fock = self._inactive_fock_matrix(frozen)

        # Calculate the inactive Fock energy
        # Only want frozen part of original OEI and inactive Fock matrix
        _oei = self.oei[np.ix_(frozen, frozen)]
        _inactive_fock = inactive_fock[np.ix_(frozen, frozen)]
        self.inactive_energy = np.einsum("ii->", _oei + _inactive_fock)

        # Keep only the active part
        self.embed_oei = inactive_fock[np.ix_(active, active)]
        self.embed_tei = self.tei[np.ix_(active, active, active, active)]

        # Update class attributes
        self.active = active
        self.frozen = frozen
        self.n_active_orbs = 2 * len(active)
        self.n_active_e = sum(self.nelec) - 2 * len(self.frozen)

    def hamiltonian(
        self,
        ham_type=None,
        oei=None,
        tei=None,
        constant=None,
        ferm_qubit_map=None,
    ):
        """
        Builds a molecular Hamiltonian using the one-/two- electron integrals

        Args:
            ham_type: Format of molecular Hamiltonian returned. The available options are:
                ``("f", "ferm")``: OpenFermion ``FermionOperator``,
                ``("q", "qubit")``: OpenFermion ``QubitOperator``, or
                ``("s", "sym")``: Qibo ``SymbolicHamiltonian`` (default)
            oei: 1-electron integrals. Default: ``self.oei`` (MO basis)
            tei: 2-electron integrals in 2ndQ notation. Default: ``self.tei`` (MO basis)
            constant: For inactive Fock energy if embedding used. Default: 0.0
            ferm_qubit_map: Which fermion to qubit transformation to use.
                Must be either ``jw`` (default) or ``bk``

        Returns:
            Molecular Hamiltonian in the format of choice
        """
        # Define default variables
        if ham_type is None:
            ham_type = "sym"
        if oei is None:
            oei = self.oei
        if tei is None:
            tei = self.tei
        if constant is None:
            constant = 0.0
        if ferm_qubit_map is None:
            ferm_qubit_map = "jw"

        constant += self.e_nuc  # Add nuclear repulsion energy

        # Start with an InteractionOperator
        ham = fermionic_hamiltonian(oei, tei, constant)
        if ham_type in ("f", "ferm"):
            # OpenFermion FermionOperator Hamiltonian
            ham = openfermion.transforms.get_fermion_operator(ham)
            ham.compress()
            return ham
        ham = qubit_hamiltonian(ham, ferm_qubit_map)
        if ham_type in ("q", "qubit"):
            # OpenFermion QubitOperator Hamiltonian
            return ham
        if ham_type in ("s", "sym"):
            # Qibo SymbolicHamiltonian
            return symbolic_hamiltonian(ham)
        # raise NameError(f"Unknown {ham_type}!")  # Shouldn't ever reach here

    @staticmethod
    def eigenvalues(hamiltonian):
        """
        Finds the lowest 6 exact eigenvalues of the molecular Hamiltonian
            Note: Uses the ``eigenvalues()`` class method for a Qibo ``SymbolicHamiltonian`` object

        Args:
            hamiltonian: Molecular Hamiltonian, given as a ``FermionOperator``, ``QubitOperator``, or
                ``SymbolicHamiltonian`` (not recommended)
        """
        if isinstance(hamiltonian, (openfermion.FermionOperator, openfermion.QubitOperator)):
            from scipy.sparse import linalg

            hamiltonian_matrix = openfermion.get_sparse_operator(hamiltonian)
            # k argument in eigsh will depend on the size of the Hamiltonian
            n_eigenvals = min(6, hamiltonian_matrix.shape[0] - 2)
            # which=SA and return_eigenvalues=False returns the eigenvalues sorted by absolute value
            eigenvalues = linalg.eigsh(hamiltonian_matrix, k=n_eigenvals, which="SA", return_eigenvectors=False)
            # So need to sort again by their (algebraic) value to get the order: smallest->largest
            return sorted(eigenvalues)
        if isinstance(hamiltonian, SymbolicHamiltonian):
            return hamiltonian.eigenvalues()
        raise TypeError("Type of Hamiltonian unknown")
