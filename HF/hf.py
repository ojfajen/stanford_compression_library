import numpy as np
from numpy import linalg as LA
import diis
from diis import diis

class Hartree_Fock(object):
    "Load 1- and 2e-integrals, then run HF."

 #   def __init__(self, enuc, s, t, v, eri, natom, znumbers, coords, ux, uy, uz):
    def __init__(self, enuc, s, hcore, eri, natom, nao, nelectron):   
        self.enuc = enuc
        self.s = s
#        self.t = t
#        self.v = v
        self.eri = eri
        self.natom = natom
        self.nelectron = nelectron
        self.nocc = int(self.nelectron / 2.0)
        self.nao = nao
        self.hcore = hcore
#        print(f'HCore: {hcore}')

#    def make_core_ham(self):
#        v = self.v
#        t = self.t
#        h_core = np.add(t, v)
        #print(h_core)
#        return h_core


    def diagonalize_overlap(self):
        s = self.s
        d, ls = LA.eigh(s)
        nd = np.power(d, -(1/2))
        lmda = np.diag(nd)
        lst = ls.transpose()
        sint = np.matmul(ls,lmda)
        s12 = np.matmul(ls, np.matmul(lmda,lst))
#        s12 = np.einsum( "ij,jk,kl", ls, lmda, lst )
#        print(f'S = {s}')
#        print(f' S-1/2 = {s12}')
#        print(f'd={d}')
#        print(f'lmda = {lmda}')
#        print(f'ls = {ls}')
#        print(f'sint = {sint}')
        return s12
        

    def get_initial_fock(self, x, hcore):
        s12 = x
        s12t = s12.transpose()
        hcore = hcore
        work = np.matmul(s12t,hcore)
#        print(f'work = {work}')
        f0 = np.matmul( s12t, np.matmul(hcore,s12))
#        f0 = np.einsum( "ij,jk,kl", s12t, hcore, s12 )
#        print(f'Transformed Fock Matrix: {f0}')
#        print(f'Orthogonalization matrix: {s12}')
 #       print(f'Initial F prime matrix: {f0}')
        return f0

    def diagonalize_fock_matrix(self, f, x):
        f0 = f
        e0, c0 = LA.eigh(f0)
#        print(f'f0 eigs: {e0}')
#        print(f'Transformed MO coeff: {c0}')
        s12 = x
        C0 = np.matmul(s12, c0)
#        print(f' Initial MO coeff: {C0}')
        return C0

    def build_initial_dm(self,c, nao):
        cua = c
        #        print(cua)
        cva = cua
        length = nao
        p = np.zeros((length, length))
        for k in range(0, length):
            row = cua[k]
            for j in range(0, length):
                column = cva[j]
                n_occ = self.nocc
                for i in range(0, n_occ):
                    ca = row[i]
                    cb = column[i]
                    paa = ca*cb
#                    print(paa)
                    p[k,j] += paa
#                paa = np.dot(row, column)
#                puv[k,j] = paa
#        print(f'Initial density matrix: {p}')
        return p


    def compound_index(self, i, j, k, l):
        if i > j:
            ij = i*(i+1)/2 + j
        else:
            ij = j*(j+1)/2 + i
        if k > l:
            kl = k*(k+1)/2 + l
        else:
            kl = l*(l+1)/2 + k
        if ij > kl:
            ijkl = ij*(ij+1)/2 + kl
        else:
            ijkl = kl*(kl+1)/2 + ij
        return ijkl
    

    def product_index(self, i, j):
        if i > j:
            ij = i*(i+1)/2 + j
        else:
            ij = j*(j+1)/2 + i
        return ij


    def compute_scf_init(self, hcore, p, f, nao):
        hcore = hcore
        p = p
        f = f
        e_nuc = self.enuc
        e_elec = 0
        length = nao
        for i in range(0, length):
            for j in range(0, length):
                p_elem = p[i,j]
                h_elem = hcore[i,j]
                fuv = h_elem
                #fuv = f[i,j]
                e_elem = p_elem * (h_elem + fuv)
                e_elec += e_elem
        #print(f'Initial electronic energy: {e_elec}')
        e_init = e_elec + e_nuc
        #print(f' Initial SCF energy: {e_init}')
        return e_init, e_elec

    
    def get_new_fock(self, hcore, p_old, nao):
        hcore = hcore
        p_old = p_old
        eri = self.eri
        nao = nao
        new_f = np.zeros((nao, nao))
        for u in range(0, nao):
            for v in range(0, nao):
                h_elem = hcore[u,v]
                g_comp = 0
                for a in range(0, nao):
                    for b in range(0, nao):
                        y, s = a+1, b+1
                        p_elem = p_old[a,b]
                        u_ind = u + 1
                        v_ind = v + 1
#                        j_cmpd = Hartree_Fock.compound_index(self, u_ind, v_ind, y, s)
#                        k_cmpd = Hartree_Fock.compound_index(self, u_ind, y, v_ind, s)
#                        j_elem = eri[int(j_cmpd)]
#                        k_elem = eri[int(k_cmpd)]
                        j_elem = eri[u,v,a,b]
                        k_elem = eri[u,a,v,b]
                        g_elem = p_elem * ( 2*j_elem - k_elem )
                        g_comp += g_elem
                f_uv = h_elem + g_comp
                new_f[u,v] = f_uv
#        print(f'New Fock matrix: {new_f}')
        return new_f


    def get_new_p(self, x, f, nao):
        s12 = x
        s12t = s12.transpose()
        f0 = f
        f2 = np.matmul(s12t, f0)
#        print(f'f2: {f2}')
        f1 = np.matmul(s12t, np.matmul(f0,s12))
#        print(f'f1: {f1}')
#        f1 = np.einsum( "ij,jk,kl", s12t, f0, s12 )
        e, c0 = LA.eigh(f1)
        c = np.matmul(s12, c0)
#        print(f'c: {c}')
        nao = nao
        p = np.zeros((nao, nao))
        for k in range(0, nao):
            row = c[k]
            for j in range(0, nao):
                column = c[j]
                n_occ = self.nocc
                for i in range(0, n_occ):
                    ca = row[i]
                    cb = column[i]
                    paa = ca*cb
#                    print(paa)
                    p[k,j] += paa
#                paa = np.dot(row, column)
#                puv[k,j] = paa
#        print(f'Initial density matrix: {p}')
        return p, c

    def compute_scf_energy(self, hcore, p, f, nao):
        hcore = hcore
        p = p
        e_nuc = self.enuc
        f = f
        nao = nao
        e_elec = 0
        for u in range(0, nao):
            for v in range(0, nao):
                puv = p[u,v]
                huv = hcore[u,v]
                fuv = f[u,v]
                e_comp = puv * (huv + fuv)
                e_elec += e_comp
        e_tot = e_elec + e_nuc
        #print(f'Current electronic energy: {e_elec}')
        #print(f'Current total energy: {e_tot}')
        return e_tot, e_elec

    
    def test_conv(self, e, e_old, p, p_old, nao):
        d_e = e - e_old
        msd = 0
        for u in range(0, nao):
            for v in range(0, nao):
                p_curr = p[u,v]
                p_last = p_old[u,v]
                sqr_diff = ( p_curr - p_last ) ** 2
                msd += sqr_diff
        rmsd = np.sqrt(msd)
        return d_e, rmsd


    def run_hf(self, e_thresh=1e-8, d_thresh=1e-8):
#        hcore = Hartree_Fock.make_core_ham(self)
        hcore = self.hcore
        x = Hartree_Fock.diagonalize_overlap(self)
        nao = self.nao
        tot_energy_list = []
        elec_list = []
        s = self.s
        #dice = diis(nao,s,x)
        iter = 0
        print('iter          electronic energy            total energy             delta E            rms D')
        if iter == 0:
            f = Hartree_Fock.get_initial_fock(self, x, hcore)
            c = Hartree_Fock.diagonalize_fock_matrix(self, f, x)
            p = Hartree_Fock.build_initial_dm(self, c, nao)
       #     print(f'Initial F matrix: {f}')
       #     print(f'Initial c matrix: {c}')
       #     print(f'Initial density matrix: {p}')
            global p_old
            p_old = p
            e = Hartree_Fock.compute_scf_init(self, hcore, p, f, nao)
            tot_energy_list.append(e[0])
            elec_list.append(e[1])
            print(f'{iter}      {e[1]}       {e[0]}')
            global old_e
            old_e = e[1]
        #    dice.run_diis(p,f)
            iter = 1
#        if iter == 1:
#            f = Hartree_Fock.get_new_fock(self, hcore, p_old, nao)
#            p, c = Hartree_Fock.get_new_p(self, x, f, nao)
#            e = Hartree_Fock.compute_scf_energy(self, hcore, p, f, nao)
#            de, rmsd = Hartree_Fock.test_conv(self, e[1], old_e, p, p_old, nao)  
#            p_old = p
#            old_e = e[1]
#            print(f'{iter}      {e[1]}     {e[0]}      {de}       {rmsd}')
#            iter = iter +1
        de = rmsd = 1.0
        while de > e_thresh or rmsd > d_thresh:
            f = Hartree_Fock.get_new_fock(self, hcore, p_old, nao) # trying something 12/7/20
        #    f = dice.run_diis(p,f) # doing everything with DIIS fock matrix
            p, c = Hartree_Fock.get_new_p(self, x, f, nao)
            e = Hartree_Fock.compute_scf_energy(self, hcore, p, f, nao) # changed to run w/ fock matrix from DIIS
            de, rmsd = Hartree_Fock.test_conv(self, e[1], old_e, p, p_old, nao)
            p_old = p
            old_e = e[1]
            global f_rec
            f_rec = f
            global c_cur
            c_cur = c
        #    print(f'Current Fock matrix: {f_rec}')
            print(f'{iter}      {e[1]}     {e[0]}      {de}       {rmsd}')
            iter = iter +1
        print(f'SCF converged with E = {e[0]} Ha')


   #     print(f'Final Fock matrix in AO basis: {f}')
   #     print(f'Final MO coeff in AO basis: {c}')
        #Now do some stuff with converged wf.
        #first, transform fock matrix to mo basis.
        p = p_old
        f = f_rec
        c = c_cur
        nao = self.nao
        f_mo = np.zeros((nao, nao))
        for i in range(0, nao):
            for j in range(0, nao):
 #               i_col = c[:,i]
 #               j_col = c[:,j]
                for u in range(0, nao):
                    for v in range(0, nao):
#                        cju = j_col[u]
#                        civ = i_col[v]
#                        fuv = f[u,v]
#                        ccf = cju * civ * fuv
                        f_mo[i,j] += c[u,i] * c[v,j] * f[u,v]
#        mo_f = np.einsum( "ij, jk, kl", c.transpose(), f, c )  
 #       print(f'Fock matrix in MO basis: {f_mo}')
 #       print(f'Fock matrix in mo basis: {mo_f}')
  #      print(f'Final MO energies: {f_mo.diagonal()}')
        return f_mo, c, e[0]
        #Now compute one-electron properties.
    
    def dipoles(self, ux, uy, uz, coords, znumbers):
        natom = int(self.natom)
        u_x = 0
        u_y = 0
        u_z = 0
        for u in range(0, nao):
            for v in range(0, nao):
                p_uv = p[u,v]
                i = int(u+1)
                j = int(v+1)
                ij = Hartree_Fock.product_index(self, i, j)
                u_uvx = ux[int(ij)]
                u_uvy = uy[int(ij)]
                u_uvz = uz[int(ij)]
                u_x += p_uv * u_uvx
                u_y += p_uv * u_uvy
                u_z += p_uv * u_uvz
        u_x = 2 * u_x
        u_y = 2 * u_y
        u_z = 2 * u_z
        for a in range(0, natom):
            zn = znum[a]
            a_coord = coords[a]
            xa = a_coord[0]
            ya = a_coord[1]
            za = a_coord[2]
            u_x += zn * xa
            u_y += zn * ya
            u_z += zn * za
#        print(f'Mu-X =     {u_x}')
#        print(f'Mu-Y =     {u_y}')
#        print(f'Mu-Z =     {u_z}')
#        print(f'Total dipole moment (au) : {np.sqrt(u_x**2 + u_y**2 + u_z**2)}')
#        return mo_f, c, e[0]
        return u_x, u_y, u_z
