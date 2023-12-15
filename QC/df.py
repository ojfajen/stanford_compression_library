from pyscf import gto, scf, mcscf, df, lib
import numpy as np
import scipy 

def mse(v1, v2):
    """
    computes the mean square error
    """
    d = np.linalg.norm(v1 - v2, ord=2)  # l2_norm
    loss = d * d / (v1.size)  # avg l2 loss
    return loss

mol = gto.M(atom='O   0.000000000000  -0.143225816552   0.000000000000; H   1.638036840407   1.136548822547  -0.000000000000; H  -1.638036840407   1.136548822547  -0.000000000000', basis='cc-pvdz')

#mol = gto.M(atom='C  -0.000000000000   0.000000000000   0.000000000000; H   1.183771681898  -1.183771681898  -1.183771681898; H   1.183771681898   1.183771681898   1.183771681898; H  -1.183771681898   1.183771681898  -1.183771681898; H  -1.183771681898  -1.183771681898   1.183771681898', basis='sto-3g')

#mol = gto.M(atom='C       0.00000000     2.62065942     0.00000000; C      -2.26955763     1.31032971     0.00000000; C      -2.26955763    -1.31032971     0.00000000; C       0.00000000    -2.62065942     0.00000000; C       2.26955763    -1.31032971     0.00000000; C       2.26955763     1.31032971     0.00000000; H      -4.04130651     2.33324940     0.00000000; H      -4.04130651    -2.33324940     0.00000000; H       0.00000000    -4.66649880     0.00000000; H       4.04130651    -2.33324940     0.00000000; H       4.04130651     2.33324940     0.00000000; H       0.00000000     4.66649880     0.00000000', basis='sto-3g') 

mol.verbose = 8
mf = scf.RHF(mol).density_fit(auxbasis='ccpvdz-jk-fit').run()    # output: -108.943773290737

auxbasis = 'ccpvdz-jk-fit'
auxmol = df.addons.make_auxmol(mol,auxbasis)
ints_3c2e = df.incore.aux_e2(mol,auxmol,intor='int3c2e')
ints_2c2e = auxmol.intor('int2c2e')

print(ints_3c2e.shape)

np.savetxt('dferi.dat',ints_3c2e.flatten())

decoded_ints = np.loadtxt('df.decoded.12bps.dat')


nao = mol.nao
naux = auxmol.nao

print("nao: ",nao)
print("naux: ",naux)

df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao,naux).T)
df_coef = df_coef.reshape(naux,nao,nao)
df_eri = lib.einsum('ijP,Pkl->ijkl', decoded_ints.reshape(nao,nao,naux), df_coef)

np.savetxt("df.12bps.full.dat",df_eri.flatten())

eri = mol.intor('int2e')

print(abs(mol.intor('int2e') - df_eri).max())
print(mse(df_eri.flatten(),eri.flatten()))
