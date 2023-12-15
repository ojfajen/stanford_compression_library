import numpy as np
import pyscf
from pyscf import gto, scf
from pyscf import ao2mo


#######################################################################
# Initialize Molecule
#######################################################################

coords = 'O   0.000000000000  -0.143225816552   0.000000000000; H   1.638036840407   1.136548822547  -0.000000000000; H  -1.638036840407   1.136548822547  -0.000000000000'


#coords = 'C -0.000000000000   0.000000000000   0.000000000000; H   1.183771681898  -1.183771681898  -1.183771681898 ;H   1.183771681898   1.183771681898   1.183771681898 ;H  -1.183771681898   1.183771681898  -1.183771681898; H  -1.183771681898  -1.183771681898   1.183771681898'

charge=0

basis = 'ccpvdz'


### Build pyscf mol object which we need to get TEIs ###
def setup(coords=coords, basis=basis):
    mol = gto.Mole()
    mol.atom = coords
    mol.basis = basis
    mol.charge = charge 
    mol.unit = 'Bohr'
    mol.build()

    return mol

### Get all integrals from pyscf ###
def get_ao_ints(mol):
    enuc = pyscf.gto.mole.energy_nuc(mol)
    nao = pyscf.gto.mole.nao_nr(mol)
    nelectron = mol.nelectron
    natm = mol.natm

    mf = scf.RHF(mol)

    s = scf.hf.get_ovlp(mol)
    hcore = scf.hf.get_hcore(mol)
    eri = mol.intor('int2e',aosym='s1')
    np.savetxt('pyscf.ints.dat',eri.flatten())


    return s, hcore, eri, enuc, nao, nelectron, natm


def mo_eri(mol,c):
    tei_mo = ao2mo.kernel(mol, c, compact=False)
    return tei_mo

mol = setup(coords,basis)
