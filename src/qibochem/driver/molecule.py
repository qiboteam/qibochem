"""
Driver for obtaining molecular integrals from either PySCF or PSI4
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openfermion
import pyscf
from qibo.hamiltonians import SymbolicHamiltonian

from qibochem.driver.hamiltonian import (
    _fermionic_hamiltonian,
    _qubit_hamiltonian,
    _qubit_to_symbolic_hamiltonian,
)


@dataclass
class Molecule:
    """
    Class representing a single molecule

    Args:
        geometry (list): Molecular coordinates in OpenFermion format,  e.g.
            ``[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))]``
        charge (int): Net electronic charge of molecule. Default: ``0``
        multiplicity (int): Spin multiplicity of molecule, given as 2S + 1, where S is half the number of unpaired
            electrons. Default: ``1``
        basis (str): Atomic orbital basis set, used for the PySCF calculations. Default: ``"STO-3G"`` (minimal basis)
        xyz_file (str): .xyz file containing the molecular coordinates. The comment line can be used to define the
            electronic charge and spin multiplity if it is given in this format: ``{charge} {multiplicity}``
        active (list): Iterable representing the set of MOs to be included in the quantum simulation
            e.g. ``list(range(3,6))`` for an active space with orbitals 3, 4 and 5.

    """

    geometry: list = None
    charge: int = 0
    multiplicity: int = 1
    basis: str = "sto-3g"
    xyz_file: str = None

    nelec: int = field(default=None, init=False)  #: Total number of electrons for the molecule
    norb: int = field(default=None, init=False)  #: Number of molecular orbitals considered for the molecule
    nso: int = field(default=None, init=False)  #: Number of molecular spin-orbitals considered for the molecule
    e_hf: float = field(default=None, init=False)  #: Hartree-Fock energy
    oei: np.ndarray = field(default=None, init=False)  #: One-electron integrals
    tei: np.ndarray = field(
        default=None, init=False
    )  #: Two-electron integrals, order follows the second quantization notation

    nalpha: int = field(default=None, init=False)  #: Number of electrons with :math:`\alpha`-spin
    nbeta: int = field(default=None, init=False)  #: Number of electrons with :math:`\beta`-spin
    e_nuc: float = field(default=None, init=False)  #: Nuclear repulsion energy for the given molecular geometry
    hcore: np.ndarray = field(default=None, init=False)
    aoeri: np.ndarray = field(default=None, init=False)
    ca: np.ndarray = field(default=None, init=False)  #: Coefficients of the Hartree-Fock molecular orbitals
    eps: np.ndarray = field(default=None, init=False)  #: Hartree-Fock orbital eigenvalues
    # Molecular properties that are currently not used/needed for anything?
    # overlap: np.ndarray = field(default=None, init=False)  #: Overlap integrals
    # ja: np.ndarray = field(default=None, init=False)  #: Coulomb repulsion between electrons
    # ka: np.ndarray = field(default=None, init=False)  #: Exchange interaction between electrons
    # fa: np.ndarray = field(default=None, init=False)  #: Fock matrix

    # For HF embedding
    active: list = None  #: Iterable of molecular orbitals included in the active space
    frozen: list = field(
        default=None, init=False
    )  #: Iterable representing the occupied molecular orbitals removed from the simulation

    inactive_energy: float = field(default=None, init=False)
    embed_oei: np.ndarray = field(default=None, init=False)
    embed_tei: np.ndarray = field(default=None, init=False)

    n_active_e: int = field(
        default=None, init=False
    )  #: Number of electrons included in the active space if HF embedding is used
    n_active_orbs: int = field(
        default=None, init=False
    )  #: Number of spin-orbitals in the active space if HF embedding is used

    # Runs after init
    def __post_init__(self):
        if self.xyz_file is not None:
            self._process_xyz_file()

    def _process_xyz_file(self):
        """
        Reads the .xyz file given when defining the Molecule to obtain the molecular coordinates (in OpenFermion
        format), charge, and multiplicity
        """
        assert Path(f"{self.xyz_file}").exists(), f"{self.xyz_file} not found!"
        with open(self.xyz_file, encoding="utf-8") as file_handler:
            # First two lines: # atoms and comment line (charge, multiplicity)
            _n_atoms = int(file_handler.readline())  # Not needed/used

            # Try to read charge and multiplicity from comment line
            split_line = [int(_num) for _num in file_handler.readline().split()]
            if len(split_line) == 2:
                # Format of comment line matches (charge, multiplicity):
                _charge, _multiplicity = split_line
                self.charge = _charge
                self.multiplicity = _multiplicity

            # Start reading xyz coordinates from the 3rd line onwards
            _geometry = []
            for line in file_handler:
                split_line = line.split()
                # OpenFermion format: [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7)), ...]
                atom_xyz = [split_line[0], tuple(float(_xyz) for _xyz in split_line[1:4])]
                _geometry.append(tuple(atom_xyz))
        self.geometry = _geometry

    def _calc_oei(self, mo_coeff):
        oei = np.einsum("ab,bc->ac", self.hcore, mo_coeff, optimize=True)
        oei = np.einsum("ab,ac->bc", mo_coeff, oei, optimize=True)
        return oei

    def _calc_tei(self, mo_coeff):
        tei = pyscf.ao2mo.kernel(self.aoeri, mo_coeff)
        tei = np.einsum("pqrs->prsq", tei, optimize=True)
        # tei = np.asarray(pyscf_mol.intor('int2e'))  # Alternative using PySCF mol directly
        # tei = np.einsum(
        #     "up, vq, uvkl, kr, ls -> prsq", mo_coeff, mo_coeff, self.aoeri, mo_coeff, mo_coeff, optimize=True
        # )  # NumPy alternative. From https://pycrawfordprogproj.readthedocs.io/en/latest/Project_04/Project_04.html
        return tei

    @property
    def ca(self):
        return self._ca

    @ca.setter
    def ca(self, new_ca):
        # Update molecular integrals when MO coefficients are updated and hcore exists
        if new_ca is not None:
            if self.hcore is not None:
                self.oei = self._calc_oei(new_ca)
            if self.aoeri is not None:
                self.tei = self._calc_tei(new_ca)
        self._ca = new_ca

    # Occupancy of molecular orbitals. Currently not needed/being used for anything?
    # @property
    # def pa(self):
    #     ca_occ = self.ca[:, : self.nalpha]
    #     return ca_occ @ ca_occ.T

    # @property
    # def da(self):
    #     ca_occ = self.ca[:, : self.nalpha]
    #     return self.ca.T @ self.overlap @ self.pa @ self.overlap @ self.ca

    def run_pyscf(self, max_scf_cycles=50):
        """
        Run a Hartree-Fock calculation with PySCF to obtain molecule quantities and molecular integrals

        Args:
            max_scf_cycles (int): Maximum number of SCF cycles in PySCF (Default: ``50``)
        """
        # Set up and run PySCF calculation
        geom_string = "".join("{} {:.6f} {:.6f} {:.6f} ; ".format(_atom[0], *_atom[1]) for _atom in self.geometry)
        spin = self.multiplicity - 1  # PySCF spin is 2S
        pyscf_mol = pyscf.gto.M(charge=self.charge, spin=spin, atom=geom_string, basis=self.basis, symmetry="C1")
        pyscf_mol.verbose = 0

        pyscf_job = pyscf.scf.RHF(pyscf_mol)
        pyscf_job.max_cycle = max_scf_cycles
        pyscf_job.run()

        # Save results from HF calculation
        self.nalpha = pyscf_mol.nelec[0]
        self.nbeta = pyscf_mol.nelec[1]
        self.nelec = sum(pyscf_mol.nelec)
        self.e_hf = pyscf_job.e_tot  # HF energy
        self.eps = np.asarray(pyscf_job.mo_energy)
        self.e_nuc = pyscf_mol.energy_nuc()
        # Alternative: hcore = np.asarray(pyscf_mol.intor("int1e_kin")) + np.asarray(pyscf_mol.intor("int1e_nuc"))
        self.hcore = pyscf_job.get_hcore()  # 'Core' (potential + kinetic) integrals
        self.aoeri = np.asarray(pyscf_mol.intor("int2e"))
        self.ca = np.asarray(pyscf_job.mo_coeff)  # MO coeffcients
        self.norb = self.ca.shape[1]
        self.nso = 2 * self.norb
        self.overlap = np.asarray(pyscf_mol.intor("int1e_ovlp"))

        # Currently unused properties?
        # self.ja = pyscf_job.get_j()
        # self.ka = pyscf_job.get_k()
        # self.fa = pyscf_job.get_fock()

    # def run_psi4(self, output=None):
    #     """
    #     Run a Hartree-Fock calculation with PSI4 to obtain the molecular quantities and
    #         molecular integrals

    #     Args:
    #         output: Name of PSI4 output file. ``None`` suppresses the output on non-Windows systems,
    #             and uses ``psi4_output.dat`` otherwise
    #     """
    #     import psi4  # pylint: disable=C0415

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
    #     e_hf, wavefn = psi4.energy("hf", return_wfn=True)
    #     mints = psi4.core.MintsHelper(wavefn.basisset())

    #     # Fill in the class attributes
    #     self.nalpha = wavefn.nalpha()
    #     self.nbeta = wavefn.nbeta()
    #     self.nelec = sum(wavefn.nalpha(), wavefn.nbeta())
    #     self.e_hf = e_hf
    #     self.eps = np.asarray(wavefn.epsilon_a())
    #     self.e_nuc = psi4_mol.nuclear_repulsion_energy()
    #     self.hcore = np.asarray(wavefn.H())
    #     self.aoeri = np.asarray(mints.ao_eri())
    #     self.ca = np.asarray(wavefn.Ca())  # MO coefficients
    #     self.norb = wavefn.nmo()
    #     self.nso = 2 * self.norb
    #     self.overlap = np.asarray(wavefn.S())

    #     self.ja = np.asarray(wavefn.jk().J()[0])
    #     self.ka = np.asarray(wavefn.jk().K()[0])
    #     self.fa = np.asarray(wavefn.Fa())

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

    def _active_space(self, active, frozen):
        """
        Helper function to check the input for active/frozen space and define the default values
        for them where necessary

        Args:
            active: Iterable representing the active-space for quantum simulation
            frozen: Iterable representing the occupied orbitals to be removed from the simulation

        Returns:
            _active, _frozen: Iterables representing the active/frozen space
        """
        n_orbs = self.norb
        n_occ_orbs = self.nalpha

        _active, _frozen = None, None
        if active is None:
            # No arguments given
            if frozen is None:
                # Default active: Full set of orbitals, frozen: empty list
                _active = list(range(n_orbs))
                _frozen = []
            # Only frozen argument given
            else:
                if frozen:
                    # Non-empty frozen space must be occupied orbitals
                    assert max(frozen) + 1 < n_occ_orbs and min(frozen) >= 0, "Frozen orbital must be occupied orbitals"
                _frozen = frozen
                # Default active: All orbitals not in frozen
                _active = [_i for _i in range(n_orbs) if _i not in _frozen]
        # active argument given
        else:
            # Check that active argument is valid
            assert max(active) < n_orbs and min(active) >= 0, "Active space must be between 0 and the number of MOs"
            _active = active
            # frozen argument not given
            if frozen is None:
                # Default frozen: All occupied orbitals not in active
                _frozen = [_i for _i in range(n_occ_orbs) if _i not in _active]
            # active, frozen arguments both given:
            else:
                # Check that active/frozen arguments don't overlap
                assert not (set(active) & set(frozen)), "Active and frozen space cannot overlap"
                if frozen:
                    # Non-empty frozen space must be occupied orbitals
                    assert max(frozen) + 1 < n_occ_orbs and min(frozen) >= 0, "Frozen orbital must be occupied orbitals"
                # All occupied orbitals have to be in active or frozen
                assert all(
                    _occ in set(active + frozen) for _occ in range(n_occ_orbs)
                ), "All occupied orbitals have to be in either the active or frozen space"
                # Hopefully no more problems with the input
                _frozen = frozen
        return _active, _frozen

    def hf_embedding(self, active=None, frozen=None):
        """
        Turns on HF embedding for a given active/frozen space, and fills in the class attributes: ``inactive_energy``
        , ``embed_oei``, and ``embed_tei``.

        Args:
            active (list): Iterable representing the active-space for quantum simulation. Uses the ``Molecule.active``
                class attribute if not given.
            frozen (list): Iterable representing the occupied orbitals to be removed from the simulation. Depends on the
                `active` argument if not given.
        """
        # Default arguments for active and frozen if no arguments given
        if active is None and frozen is None:
            _active, _frozen = self._active_space(self.active, self.frozen)
        else:
            # active/frozen arguments given, process them using _active_space similarly
            _active, _frozen = self._active_space(active, frozen)
        # Update the class attributes with the checked arguments
        self.active = _active
        self.frozen = _frozen

        # Build the inactive Fock matrix first
        inactive_fock = self._inactive_fock_matrix(self.frozen)

        # Calculate the inactive Fock energy
        # Only want frozen part of original OEI and inactive Fock matrix
        _oei = self.oei[np.ix_(self.frozen, self.frozen)]
        _inactive_fock = inactive_fock[np.ix_(self.frozen, self.frozen)]
        self.inactive_energy = np.einsum("ii->", _oei + _inactive_fock)

        # Keep only the active part
        self.embed_oei = inactive_fock[np.ix_(self.active, self.active)]
        self.embed_tei = self.tei[np.ix_(self.active, self.active, self.active, self.active)]

        # Update other class attributes
        self.n_active_orbs = 2 * len(self.active)
        self.n_active_e = self.nelec - 2 * len(self.frozen)

    @staticmethod
    def _filter_array(array, threshold):
        """Helper function to filter out v. small values from an array"""
        cp_array = np.copy(array)
        cp_array[np.abs(array) < threshold] = 0.0
        return cp_array

    def hamiltonian(
        self,
        ham_type=None,
        oei=None,
        tei=None,
        constant=None,
        ferm_qubit_map=None,
        threshold=1e-12,
    ):
        """
        Builds a molecular Hamiltonian using the one-/two- electron integrals. If HF embedding has been applied,
        (i.e. the ``embed_oei``, ``embed_tei``, and ``inactive_energy`` class attributes are all not ``None``), the
        corresponding values for the molecular integrals will be used instead.

        Args:
            ham_type (str): Format of molecular Hamiltonian returned. The available options are:
                ``("f", "ferm")``: :class:`openfermion.FermionOperator`,
                ``("q", "qubit")``: :class:`openfermion.QubitOperator`, or
                ``("s", "sym")``: :class:`qibo.hamiltonians.SymbolicHamiltonian` (default)
            oei (ndarray): 1-electron integrals (in the MO basis). The default value is the ``oei`` class attribute,
                unless the ``embed_oei`` attribute exists and is not ``None``, then ``embed_oei`` is used.
            tei (ndarray): 2-electron integrals in the second-quantization notation (and MO basis). The default value
                is the ``tei`` class attribute , unless the ``embed_tei`` attribute exists and is not ``None``, then
                ``embed_tei`` is used.
            constant (float): Constant value to be added to the electronic energy. Mainly used for adding the inactive
                Fock energy if HF embedding was applied. Default: 0.0, unless the ``inactive_energy`` class attribute
                exists and is not ``None``, then ``inactive_energy`` is used.
            ferm_qubit_map (str): Which fermion to qubit transformation to use. Must be either ``"jw"`` (Default)
                or ``"bk"``
            threshold (float): Threshold at which the elements of ``oei``/``tei`` are ignored, i.e. set to 0.0.
                Default: ``1e-12``

        Returns:
            :class:`openfermion.FermionOperator` or :class:`openfermion.QubitOperator`
            or :class:`qibo.hamiltonians.SymbolicHamiltonian`: Molecular Hamiltonian in the format of choice
        """
        # Define default variables
        if ham_type is None:
            ham_type = "sym"
        if oei is None:
            oei = self.oei if self.embed_oei is None else self.embed_oei
        if tei is None:
            tei = self.tei if self.embed_tei is None else self.embed_tei
        if constant is None:
            constant = 0.0 if self.inactive_energy is None else self.inactive_energy
        if ferm_qubit_map is None:
            ferm_qubit_map = "jw"

        constant += self.e_nuc  # Add nuclear repulsion energy

        # Filter out v. small values
        oei = self._filter_array(oei, threshold)
        tei = self._filter_array(tei, threshold)

        # Start with an InteractionOperator
        ham = _fermionic_hamiltonian(oei, tei, constant)
        if ham_type in ("f", "ferm"):
            # OpenFermion FermionOperator Hamiltonian
            ham = openfermion.transforms.get_fermion_operator(ham)
            ham.compress()
            return ham
        ham = _qubit_hamiltonian(ham, ferm_qubit_map)
        if ham_type in ("q", "qubit"):
            # OpenFermion QubitOperator Hamiltonian
            return ham
        if ham_type in ("s", "sym"):
            # Qibo SymbolicHamiltonian
            return _qubit_to_symbolic_hamiltonian(ham)
        raise NameError(f"Unknown {ham_type}!")  # Shouldn't ever reach here

    @staticmethod
    def eigenvalues(hamiltonian):
        """
        Finds the lowest 6 exact eigenvalues of a given Hamiltonian

        Args:
            hamiltonian (:class:`openfermion.FermionOperator` or :class:`openfermion.QubitOperator` \
            or :class:`qibo.hamiltonians.SymbolicHamiltonian`):
                Hamiltonian of interest. If the input is a :class:`qibo.hamiltonians.SymbolicHamiltonian`, the whole
                Hamiltonian matrix has to be built first (not recommended).
        """
        if isinstance(hamiltonian, (openfermion.FermionOperator, openfermion.QubitOperator)):
            from scipy.sparse import linalg  # pylint: disable=C0415

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
