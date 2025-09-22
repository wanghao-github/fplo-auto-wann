# main.py
import numpy as np
from read_hamdata import TBHamiltonian
from green_function import GreenFunction
# file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\fplohamdatatest\Fe\fplo22\Fe_4sp_3d_cutoff_25\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\Works\Orbital Hall\fcc Pt\FPLO\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\SHC and OHC from FPLO\Pt\20241029\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\SHC and OHC from FPLO\MoS2\fplo_auto_mode_all\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\Works\SymmJij\NiO\FPLO_wannier\x.wann_Ni_4s_3d_4p_O_2s_2p\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\2024work\srmn2bi2\afm_fplo_automode_all\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\SHC and OHC from FPLO\MoS2\FPLO_OHC_Mo_d_S_p\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\fplohamdatatest\graphene\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\SHC and OHC from FPLO\fcc Al\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\fplohamdatatest\mos2\new_2024_10_21\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\fplohamdatatest\bi\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\SHC and OHC from FPLO\Bi\FPLO\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\SHC and OHC from FPLO\MoS2\mos2y.fploz.fplo_no_Mo_s\\'
# file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\fplohamdatatest\SrMn2Bi2\\'
# file_path = '/Users/haowang/Library/CloudStorage/OneDrive-Personal/Desktop/Codes/fplohamdatatest/NiO//'
# file_path = '/home/hw86gixa/hw86gixa/1.NiO/x.test_FPLO_hr_tb2j/a.P1_symm/wann//'
# file_path = '/home/hw86gixa/hao_project/test/fplo_test/fe/v.wann_Fe_sd//'
# file_path = '/home/hw86gixa/hw86gixa/test/graphene//'
#file_path = '/home/hw86gixa/hw86gixa/test/graphene/z.all_modes//'
#file_path = '/home/hw86gixa/hw86gixa/test/graphene/y.test_R_r//'
#file_path = '/home/hw86gixa/hw86gixa/mos2//'
file_path  ='/home/hw86gixa/hw86gixa/y.ptbi2/z.test_win_max/emax_2.4/x.correct_round_floor//'
#file_path = '/home/hw86gixa/hw86gixa/test/bi/z.fplo_wann_bi_s_p//'
#file_path = '/home/hw86gixa/hw86gixa/w.ohc_HTP/x.ohc/output/Pt//'

Ham = TBHamiltonian(file_path)

#GreenFunction
kpath, band_wanbandtb, kdist = Ham.read_wanbandtb()

Ham.save_to_wannier_hr()
Ham.save_to_rspauli()
Ham.save_to_orbital_inp()
Ham.save_to_orbital_inp2()
Ham.save_to_nrpts_inp()
Ham.save_to_hopping()
Ham.save_to_ahe_inp()

num_wann, ndegen, irvec, HmnR_np_iR = Ham.read_wannier90_hr()
#k = np.array([0,0,0])
#Ham.get_ham_k(k)


band_structure = Ham.get_bands_from_ham(kpath, irvec, ndegen, Ham.lattice_vectors, Ham.lattice_moduli, HmnR_np_iR)


Ham.plot_bands(kdist, band_wanbandtb, band_structure)
#print("Ham.lattice_vectors:",Ham.lattice_vectors)
#print("Ham.lattice_moduli:", Ham.lattice_moduli)
Ham.save_to_wannier90_centres_xyz()
Ham.save_to_wannier_hr()

# Ham.assembly_ham(kpath, irvec, ndegen, Ham.lattice_vectors, Ham.lattice_moduli, HmnR_np_iR)
# Ham.pauli_block_all()
