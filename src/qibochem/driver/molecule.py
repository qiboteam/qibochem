import numpy as np
import openfermion as of 

import pyscf


class molecule():
    
    def __init__(self, of_geometry, charge=0, multiplicity=1, rhf=None):
        """
        Args:
            geometry: molecular geometry in openfermion format
                e.g. [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))]
        """
        self.of_geometry = of_geometry
        self.charge = charge
        self.multiplicity = multiplicity
        if rhf == None: 
            if multiplicity == 1:
                rhf = True
            else: 
                rhf = False
        self.rhf = rhf
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
        
        
    def run_pyscf(self, basis='sto-3g', output='pyscf_output.out'):
        
        geom_string = "".join("{} {:.6f} {:.6f} {:.6f} ; ".format(_atom[0], *_atom[1]) for _atom in self.of_geometry)
        spin = (self.multiplicity-1) //2 
        pyscf_mol = pyscf.gto.M(charge=self.charge, spin=spin, atom=geom_string, basis=basis, symmetry='C1')
        pyscf_job = pyscf.scf.RHF(pyscf_mol).run()
        #print(dir(pyscf_job))
        
        ehf = pyscf_job.e_tot

        ca = np.asarray(pyscf_job.mo_coeff)


        
        oei = np.asarray(pyscf_mol.intor('int1e_kin')) + np.asarray(pyscf_mol.intor('int1e_nuc'))
        #oei = np.asarray(pyscf_mol.get_hcore())
        oei = np.einsum("ab,bc->ac", oei, ca)
        oei = np.einsum("ab,ac->bc", ca, oei)
                
        #tei = np.asarray(pyscf_mol.intor('int2e'))
        eri = pyscf.ao2mo.kernel(pyscf_mol, ca)
        eri4 = pyscf.ao2mo.restore('s1', eri, ca.shape[1])
       
        tei = np.einsum("pqrs->prsq", eri4)

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
        if self.rhf == True:
            self.cb = ca
            self.db = da
            #self.fb = fa

            

    def get_ofdata(self):
        '''
        Creates an openfermion MolecularData object from pyscf or psi4 data
        '''
        
        if self.e_hf == None:
            raise Exception('Run pyscf or psi4 first to obtain molecular data.')

        # create openfermion MolecularData object
        molecule_data = of.chem.MolecularData(self.of_geometry, basis=self.basis, 
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
