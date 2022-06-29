from pyscf import gto, scf
import numpy as np

class pyscf_molecule():

    def __init__(self, atom, charge=0, spin=0, basis='sto-3g', symmetry='C1'):
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.symmetry = symmetry
        self.mol = gto.M(
            charge = charge,
            spin = spin,
            atom = atom,
            basis = basis,
            symmetry = symmetry)
                
    def run(self, method='rhf'):
        if method=='rhf':
            job = scf.RHF(self.mol).run()
            self.c = np.asarray(job.mo_coeff)
            self.ca = self.c
            self.cb = self.c
            self.energy = job.e_tot
            
        self.nbf = self.mol.nao_nr()
        self.nalpha = self.mol.nelectron//2 + self.mol.spin
        self.nbeta = self.mol.nelectron//2 - self.mol.spin
        self.overlap = np.asarray(self.mol.intor('int1e_ovlp'))
        self.t = np.asarray(self.mol.intor('int1e_kin'))
        self.v = np.asarray(self.mol.intor('int1e_nuc'))
        self.hcore = self.t + self.v
        self.aoeri = np.asarray(self.mol.intor('int2e'))
        self.enuc = self.mol.energy_nuc()
