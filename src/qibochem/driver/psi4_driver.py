import psi4
import numpy as np

class psi4_driver():
    
    def __init__(self, geometry, opts):
        
        self.geometry = geometry
        self.opts = opts
        
        
    def run(self, output='output.dat', method='scf', memory='500 MB'):
        molecule = psi4.geometry(self.geometry)
        opts = self.opts
        psi4.core.clean()
        psi4.set_memory(memory)
        psi4.set_options(opts)
        psi4.core.set_output_file(output, True)
        E, wfn = psi4.energy(method, return_wfn=True)
        #print('Energy: ', E)
        self.energy = E
        self.wfn = wfn
        
        mints = psi4.core.MintsHelper(wfn.basisset())
        self.nbf = mints.nbf()
        self.norb = wfn.nmo()
        self.nalpha = wfn.nalpha()
        self.nbeta = wfn.nbeta()
        self.hcore = np.asarray(wfn.H())
        self.aoeri = np.asarray(mints.ao_eri())
        self.ca = np.asarray(wfn.Ca())
        self.cb = np.asarray(wfn.Cb())
        self.overlap = np.asarray(wfn.S())
        self.pa = np.asarray(wfn.Da())
        self.pb = np.asarray(wfn.Db())
        self.fa = np.asarray(wfn.Fa()) # AO Fock
        self.fb = np.asarray(wfn.Fb())
        self.enuc = molecule.nuclear_repulsion_energy()
