import numpy as np
import pyscf
from pyscf import gto, scf
from pyscf import ao2mo

#coords = 'H 0.0 0.0 0.0; H 0.0 0.0 1.0'

#coords = 'O 0.000000000000  -0.143225816552   0.000000000000; H 1.638036840407   1.136548822547  -0.000000000000; H -1.638036840407   1.136548822547  -0.000000000000' 


coords = 'O   0.000000000000  -0.143225816552   0.000000000000; H   1.638036840407   1.136548822547  -0.000000000000; H  -1.638036840407   1.136548822547  -0.000000000000'

#coords = 'S -0.22044281  0.0425091   0.12670914; F -0.39385246  3.15394654 -0.04671109; F 0.08501846  0.04250572 -2.80386899; F -3.1510216   0.04250572  0.4321675; F -0.39385406 -3.06892787 -0.04671269;'   

#coords = 'C -0.000000000000   0.000000000000   0.000000000000; H   1.183771681898  -1.183771681898  -1.183771681898 ;H   1.183771681898   1.183771681898   1.183771681898 ;H  -1.183771681898   1.183771681898  -1.183771681898; H  -1.183771681898  -1.183771681898   1.183771681898'

#coords = 'H 0.0 0.0 0.0; H 0.0 0.0 1'

#coords = 'H 0.0 0.0 0.0; Cl 0.0 0.0 1.27857'

#coords = [['H', ( 0.0,   0.,  0.0)],
#          ['Cl', ( 0.0,   0., bond_dist)],

#coords = 'H 0.0 0.0 0.0; C 0.0 0.0 1.03778578; N 0.0 0.0 2.20441524'
charge=0

#coords = 'H 0.0 0.0 0.0; F 0.0 0.0 -1.11360383; F 0.0 0.0 1.11360383'
#charge = -1

basis = 'ccpvdz'

def setup(coords=coords, basis=basis):
    mol = gto.Mole()
    mol.atom = coords
    mol.basis = basis
    mol.charge = charge 
    mol.unit = 'Bohr'
    mol.build()

    return mol

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
    #tei_mo = ao2mo.restore(1,mol.intor('int2e'),mol.nao_nr())
    return tei_mo

mol = setup(coords,basis)
