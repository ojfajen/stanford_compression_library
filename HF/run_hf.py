import numpy as np
import hf
from hf import Hartree_Fock
import mp2
import control
from control import setup
from control import get_ao_ints
from control import mo_eri

mol = setup()

s, hcore, eri_array, enuc, nao, nelectron, natm = get_ao_ints(mol)
eri_array = np.loadtxt("df.1bps.full.dat")
eri_array = eri_array.reshape(nao,nao,nao,nao)
scf = hf.Hartree_Fock(enuc, s, hcore, eri_array, natm, nao, nelectron)

print(nao)

f, c, e = scf.run_hf(e_thresh=1e-8, d_thresh=1e-8)


