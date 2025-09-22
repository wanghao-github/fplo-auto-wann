import numpy as np
import matplotlib.pyplot as plt
import re

class TBHamiltonian:
    def __init__(self, file_path):
        
        self.file_path = file_path
        self.Ham_bulk = None
        self.nwan, self.lattice_vectors, self.centering, self.conventional_matrix, self.lattice_moduli,self.tij_hij_blocks, self.wancenters, self.block_lengths, self.p_wfpairs, self.wannames,self.wannames_with_n= self.read_hamdata()
        # self.nrpts, self.assembled_ham_data,self.assembled_spin_ham_data, self.unique_r_vectors, self.r_vector_count_list = self.fplo_hop_block_to_wann_hr()
        self.nrpts, self.assembled_ham_data,self.assembled_spin_ham_data, self.unique_r_vectors,self.r_vector_count_list = self.optimized_fplo_hop_block_to_wann_hr()
        # self.MI, self.Mx, self.My, self.Mz = self.pauli_block_all() 
        self.irvec = None
        self.ndegen = None
        self.HmnR_np_iR = None
        self.Ham_bulk_k = None

    def gen_kpath(self):
        full_path = self.file_path + 'syml'
        with open(full_path, 'r') as f:
            lines = f.readlines()
        num_high_sym_points = int(lines[0].strip())
        num_kpts_on_path_values = list(map(int, lines[1].strip().split()))
        num_kpts_on_path = np.array(num_kpts_on_path_values)
        high_sym_point_dict = {}
        k_symbol = []

        for i in range(num_high_sym_points):
            key = str(lines[2 + i].strip()[0])
            value = np.array(list(map(float, (lines[2 + i].strip()[1:].split()))))
            high_sym_point_dict[key] = value
            k_symbol.append(key)
        num_high_sym_path = num_high_sym_points - 1
        kpath = {}
        for i in range(num_high_sym_path):
            kpath[i] = np.zeros((num_kpts_on_path[i], 3))
        for i in range(num_high_sym_path):
            for j in range(num_kpts_on_path[i]):
                for k in range(3):
                    if i != range(num_high_sym_path):
                        kpath[i][j][k] = j * ((high_sym_point_dict[k_symbol[i + 1]][k] - high_sym_point_dict[k_symbol[i]][k]) / num_kpts_on_path[i]) + high_sym_point_dict[k_symbol[i]][k]
        kpoint_array = np.vstack([kpath[key] for key in sorted(kpath.keys())])
        return kpoint_array

    # def monkhorst_pack(size,gamma_cnenter=False):
    #     kpts = np.indices(size).transpose((1,2,3,0)).reshape((-1,3))
    #     asize = np.array(size)
    #     shift = 

    def read_wanbandtb(self):
        full_path = self.file_path + '+wanbandtb'
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            kpoints_line = lines[1::2]
            energy_line = lines[2::2]
            kpts_num = len(kpoints_line)
            bands_num = len(energy_line[0].split()) - 1
            kpoints = np.zeros((kpts_num, 3))
            bands = np.zeros((kpts_num, bands_num))
            kdist = np.zeros(kpts_num)

            for i, line in enumerate(kpoints_line):
                values = line.split()
                for j in range(3):
                    kpoints[i][j] = float(values[j + 1])
            for k, line in enumerate(energy_line):
                values = line.split()
                kdist[k] = values[0]
                for n in range(bands_num):
                    bands[k][n] = float(values[n + 1])
            return kpoints, bands, kdist

        except FileNotFoundError:
            print(f"File not found: {full_path}")
            return None, None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None

    def read_hamdata(self):
        full_path = self.file_path + '+hamdata'
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        orbital_map = {'s': 0, 'p': 1, 'd': 2,'f':3}
        spin_map = {'up': 1, 'dn': 2}
        pattern = re.compile(r"(\D+)(\d+)\s(\d+)([spdf])([+-]\d+)(up|dn)")
        
        nwan_value = None
        lattice_vectors = []
        tij_hij_blocks = []
        wannames = []
        wannames_with_n = []
        wancenters = []
        centering = []
        block_lengths = []
        p_wfpairs_list = []
        for i, line in enumerate(lines):
            if 'nwan:' in line:
                if i + 1 < len(lines):
                    nwan_value = int(lines[i + 1].strip())

            if 'lattice_vectors:' in line:
                for j in range(1, 4):
                    if i + j < len(lines):
                        vector = list(map(float, lines[i + j].strip().split()))
                        lattice_vectors.append(vector)
                        
            if 'wannames:' in line:
                for j in range(1, int(nwan_value) + 1):
                    if i + j < len(lines):
                        orb_nlm = str(lines[i + j])
                        match = pattern.match(orb_nlm)
                        if match:
                            atom_num = int(match.group(2))
                            orb_type_n = int(match.group(3))
                            orb_type_l = orbital_map[match.group(4)]
                            anglmom_number_m = int(match.group(5))
                            spin = spin_map[match.group(6)]                      
                        wannames.append([atom_num, j,orb_type_l, anglmom_number_m, spin])
                        wannames_with_n.append([atom_num, j,orb_type_n,orb_type_l, anglmom_number_m, spin])
                        
                        
            if 'centering:' in line:
                for j in range(1, 4):
                    if i + j < len(lines):
                        center = list(map(float, lines[i + j].strip().split()))
                        centering.append(center)
            
            if 'wancenters:' in line:
                for j in range(1, int(nwan_value) + 1):
                    if i + j < len(lines):
                        wancenter = list(map(float, lines[i + j].strip().split()))
                        wancenters.append(wancenter)

            if 'Tij, Hij, Sij:' in line:
                block = []
                try:
                    index1, index2 = map(int, lines[i + 1].strip().split())

                    for j in range(2, len(lines) - i):
                        if 'end Tij, Hij:' in lines[i + j]:
                            break
                        try:
                            block.append(list(map(float, lines[i + j].strip().split())))
                        except ValueError as e:
                            continue
                    if block:
                        tij_hij_blocks.append((index1, index2, block))
                        block_lengths.append((index1, index2, len(block)))
                        for irs in range(len(block)):
                            p_wfpairs_list.append((block[irs][:3], index1, index2, block[irs][3], block[irs][4], irs))
                except ValueError as e:
                    continue

        lattice_vectors = np.array(lattice_vectors)
        
        centering =np.array(centering)
        conventional_matrix = np.dot(np.linalg.inv(centering),lattice_vectors)
        lattice_moduli = np.linalg.norm(conventional_matrix, axis=1)
        p_wfpairs = np.array(p_wfpairs_list, dtype=object)
        return nwan_value, lattice_vectors, centering, conventional_matrix,lattice_moduli, tij_hij_blocks, wancenters, block_lengths, p_wfpairs,wannames,wannames_with_n


    def optimized_fplo_hop_block_to_wann_hr(self):
        unique_r_vectors = []
        inv_lattice_vectors = np.linalg.inv(self.lattice_vectors).T
        data_dict = {}
        spin_data_dict = {}
        r_vector_count = {}

        # 收集所有可能的 T_int 值
        for (index1, index2, block) in self.tij_hij_blocks:
            si = np.array(self.wancenters[index1 - 1])
            sj = np.array(self.wancenters[index2 - 1])
            si_frac = np.dot(inv_lattice_vectors, si)
            si_in_which_unitcell = np.floor(si_frac).astype(int)
            for row in block:
                r = np.array(row[:3])
                R = r + si
                R_frac = np.dot(inv_lattice_vectors, R)
                T_int = np.floor(R_frac).astype(int) - si_in_which_unitcell
                unique_r_vectors.append(T_int[:3])

        # 确定唯一的 (r1, r2, r3) 组合
        unique_r_vectors = np.unique(unique_r_vectors, axis=0)

        # 初始化 data_dict 和 spin_data_dict
        for r1, r2, r3 in unique_r_vectors:
            for i in range(1, self.nwan + 1):
                for j in range(1, self.nwan + 1):
                    data_dict[(r1, r2, r3, i, j)] = (0.0, 0.0)
                    for spin_dir in range(1, 4):
                        spin_data_dict[(r1, r2, r3, i, j, spin_dir)] = (0.0, 0.0)

        # 填充实际数据
        for (index1, index2, block) in self.tij_hij_blocks:
            si = np.array(self.wancenters[index1 - 1])
            sj = np.array(self.wancenters[index2 - 1])

            si_frac = np.dot(inv_lattice_vectors, si)
            si_in_which_unitcell = np.floor(si_frac).astype(int)
            
            for row in block:    
                r = np.array(row[:3])
                R = r + si
                R_frac = np.dot(inv_lattice_vectors, R)
                T_int = np.floor(R_frac).astype(int) - si_in_which_unitcell
                
                key = (T_int[0], T_int[1], T_int[2], index1, index2)
                data_dict[key] = (
                    data_dict[key][0] + row[3], 
                    data_dict[key][1] + row[4]
                )

                for spin_dir in range(1, 4):
                    spin_key = (T_int[0], T_int[1], T_int[2], index1, index2, spin_dir)
                    spin_data_dict[spin_key] = (
                        spin_data_dict[spin_key][0] + row[2 * spin_dir + 3],
                        spin_data_dict[spin_key][1] + row[2 * spin_dir + 4]
                    )

                r_vector_count[key] = r_vector_count.get(key, 0) + 1

        # 统计唯一 r-vectors 数量和转换字典格式
        nrpts = unique_r_vectors.shape[0]
        r_vector_count_list = sorted(r_vector_count.items())
        assembled_ham_data = [(*k, *v) for k, v in data_dict.items()]
        assembled_spin_ham_data = [(*k, *v) for k, v in spin_data_dict.items()]

        return nrpts, assembled_ham_data, assembled_spin_ham_data, unique_r_vectors, r_vector_count_list




    def fplo_hop_block_to_wann_hr(self):
        unique_r_vectors = []
        inv_lattice_vectors = np.linalg.inv(self.lattice_vectors)
        r_vector_count = {}

        for (index1, index2, block) in self.tij_hij_blocks:
            si = np.array(self.wancenters[index1 - 1])
            sj = np.array(self.wancenters[index2 - 1])
            si_frac = np.dot(inv_lattice_vectors, si)
            si_in_which_unitcell = np.floor(si_frac).astype(int)
            for row in block:
                #r = np.array(row[:3])
                #si = np.array(self.wancenters[index1 - 1])
                #sj = np.array(self.wancenters[index2 - 1])
                ##R = r - (sj - si)
                #R = r - si
                #T = np.dot(R, inv_lattice_vectors)
                #T_rounded = np.rint(T)
                #T_int = T_rounded.astype(int)
                #unique_r_vectors.append((T_int[0], T_int[1], T_int[2], index1, index2))
                r = np.array(row[:3])
                R = r + si
                R_frac = np.dot(inv_lattice_vectors, R)
                T_int = np.floor(R_frac).astype(int) - si_in_which_unitcell
                unique_r_vectors.append(T_int[:3])
        
        #unique_r_vectors = np.array(unique_r_vectors)
        unique_r_vectors = np.unique(unique_r_vectors, axis=0)

        print("len  unique_r_vectors",len(unique_r_vectors))
        data_dict = {}
        spin_data_dict = {}
        for r_vector in unique_r_vectors:
            r1, r2, r3 = r_vector[:3]
            for i in range(1, self.nwan + 1):
                for j in range(1, self.nwan + 1):
                    # print("j",j)
                    data_dict[(r1, r2, r3, i, j)] = (0.0, 0.0)
                    for k in range(1,4):
                        # print("k",k)
                        spin_data_dict[(r1, r2, r3, i, j, k)] = (0.0, 0.0)

        for (index1, index2, block) in self.tij_hij_blocks:
            si = np.array(self.wancenters[index1 - 1])
            sj = np.array(self.wancenters[index2 - 1])
            si_frac = np.dot(inv_lattice_vectors, si)
            si_in_which_unitcell = np.floor(si_frac).astype(int)
            for row in block:
                r = np.array(row[:3])
                R = r + si
                R_frac = np.dot(inv_lattice_vectors, R)
                T_int = np.floor(R_frac).astype(int) - si_in_which_unitcell
                unique_r_vectors.append(T_int[:3])
                
                #r = np.array(row[:3])
                #si = np.array(self.wancenters[index1 - 1])
                #sj = np.array(self.wancenters[index2 - 1])
                #R = r - (sj - si)
                #R = r - si
                #T = np.dot(R, inv_lattice_vectors)
                #T_rounded = np.rint(T)
                #T_int = T_rounded.astype(int)
                real = row[3]
                img = row[4]
                # spin_real_x = row[5]
                # spin_real_y = row[6]
                # spin_real_z = row[7]
                # spin_img_x = row[8]
                # spin_img_y = row[9]
                # spin_img_z = row[10]
                key = (T_int[0], T_int[1], T_int[2], index1, index2)
                if key in data_dict:
                    print("existing!!")
                    existing_real, existing_img = data_dict[key]
                    data_dict[key] = (existing_real + real, existing_img + img)
                else:
                    data_dict[key] = (real, img)
                
                for spin_dir in range(1,4):
                    spin_key = (T_int[0], T_int[1], T_int[2], index1, index2, spin_dir)
                    if spin_key in spin_data_dict:
                        existing_spin_real, existing_spin_img = spin_data_dict[spin_key]
                        spin_data_dict[spin_key] = (existing_spin_real + row[2*spin_dir+3], existing_spin_img + row[2*spin_dir+4])
                    else:
                        spin_data_dict[spin_key] = (row[2*spin_dir+3],row[2*spin_dir+4])
        assembled_ham_data = [(*k, *v) for k, v in data_dict.items()]
        assembled_spin_ham_data = [(*k, *v) for k, v in spin_data_dict.items()]
        r_vector_count_list = sorted(r_vector_count.items())
        unique_r_vectors_set = set(map(tuple, unique_r_vectors[:, :3]))
        nrpts = len(unique_r_vectors_set)
        return nrpts, assembled_ham_data, assembled_spin_ham_data, unique_r_vectors, r_vector_count_list

    def save_to_wannier90_centres_xyz(self):
        full_path = self.file_path + 'wannier90_centres.xyz'
        header = "Generated by Hao Wang's script\n"
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(f"{self.nwan}\n")
            f.write(header)
            for line in self.wancenters:
                f.write(
                    f"X          {line[0]:18.8f} {line[1]:18.8f} {line[2]:18.8f}\n"
                )
            f.write(f"Bi         2.367314417586326     -4.100308848749840     29.667930445390219 \n")
            f.write(f"Bi        -2.367314417586326      4.100308848749840    -29.667930445390219 \n")

    def save_to_wannier_hr(self):
        full_path = self.file_path + 'wannier90_hr.dat'
        header = "Generated by Hao Wang's script\n"
        sorted_ham_data = sorted(self.assembled_ham_data, key=lambda x: (x[0], x[1], x[2]))
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(f"{self.nwan}\n")
            f.write(f"{self.nrpts}\n")

            ones_line = '    1' * 15
            for _ in range(self.nrpts // 15):
                f.write(f"{ones_line}\n")
            if self.nrpts % 15 > 0:
                f.write(f"{'    1' * (self.nrpts % 15)}\n")
            for line in sorted_ham_data:
                f.write(
                    f"{line[0]:5.0f} {line[1]:5.0f} {line[2]:5.0f} {int(line[3]):5d} {int(line[4]):5d} {line[5]:18.8f} {line[6]:18.8f}\n"
                )
                
    def save_to_rspauli(self):
        full_path = self.file_path + 'rspauli.1'
        sorted_spin_ham_data = sorted(self.assembled_spin_ham_data, key=lambda x: (x[0], x[1], x[2], x[3]))
        with open(full_path, 'w', encoding='utf-8') as f:
            for line in sorted_spin_ham_data:
                f.write(
                    f"{line[0]:5.0f} {line[1]:5.0f} {line[2]:5.0f} {int(line[3]):5d} {int(line[4]):5d} {int(line[5]):5d} {line[6]:18.8f} {line[7]:18.8f}\n"
                )
    
    def save_to_hopping(self):
        full_path = self.file_path + 'hopping.1'
        sorted_ham_data = sorted(self.assembled_ham_data, key=lambda x: (x[0], x[1], x[2], x[3]))
        with open(full_path, 'w', encoding='utf-8') as f:
            for line in sorted_ham_data:
                f.write(
                    f"{line[0]:5.0f} {line[1]:5.0f} {line[2]:5.0f} {int(line[3]):5d} {int(line[4]):5d} {line[5]:18.8f} {line[6]:18.8f}\n"
                )
                 
    def save_to_orbital_inp(self):            
        full_path = self.file_path + 'orbital_inp'
        with open(full_path, 'w', encoding='utf-8') as f:
            for line in self.wannames:
                f.write(
                    f"{line[0]:5d} {line[1]:5d} {line[2]:5d} {int(line[3]):5d} {int(line[4]):5d}\n"
                )
    
    def save_to_orbital_inp2(self):            
        full_path = self.file_path + 'orbital_inp2'
        with open(full_path, 'w', encoding='utf-8') as f:
            for line in self.wannames_with_n:
                f.write(
                    f"{line[0]:5d} {line[1]:5d} {line[2]:5d} {int(line[3]):5d} {int(line[4]):5d} {int(line[5]):5d} \n"
                )
    
    
    def save_to_ahe_inp(self):            
        full_path = self.file_path + 'ahe_inp'
        with open(full_path, 'w', encoding='utf-8') as f:
            for row in self.conventional_matrix:
                # f.write(" ".join(map(str, line))+"\n")
                f.write(" ".join(f"{num:18.12f}" for num in row)+"\n")
            f.write(
                "-12  12 100  ! fermi_min,fermi_max,num_steps\n"
                "200  200  1  ! grid\n"
                "3            ! maxdim \n"
                "22           ! occupation_number\n"
                "300          ! temperature\n"
                "f            ! \n"
                "1.0 1.0      ! lambda_p lambda_d\n"
            )

                
    def save_to_nrpts_inp(self):           
        full_path = self.file_path + 'nrpts_inp'
        with open(full_path, 'w', encoding='utf-8') as f:
            # f.write(f"{self.nrpts}\n")
            ones_line = '    1' * 15
            for _ in range(self.nrpts // 15):
                f.write(f"{ones_line}\n")
            if self.nrpts % 15 > 0:
                f.write(f"{'    1' * (self.nrpts % 15)}\n")
                
    def read_wannier90_hr(self):
        full_path = self.file_path + 'wannier90_hr.dat'
        with open(full_path, 'r') as f:
            lines = f.readlines()
            num_wann = int(lines[1])
            nrpts = int(lines[2])
            lines_nrpts = int(np.ceil(nrpts / 15.0))
            ndegen_list = []
            for i in range(lines_nrpts):
                for j in range(len(lines[3 + i].split())):
                    ndegen_list.append(int(lines[3 + i].split()[j]))
            ndegen = np.array(ndegen_list)
            HmnR_np = np.zeros((num_wann ** 2 * nrpts, 7))
            for i in range(num_wann ** 2 * nrpts):
                for j in range(7):
                    HmnR_np[i][j] = (float(lines[3 + lines_nrpts + i].split()[j]))
            HmnR_np_iR = np.zeros((num_wann, num_wann, nrpts), dtype=complex)
            irvec = np.zeros((nrpts, 3))
            for ir in range(nrpts):
                for n in range(num_wann):
                    for m in range(num_wann):
                        HmnR_np_iR[m, n, ir] = complex(HmnR_np[ir * num_wann ** 2 + n * num_wann + m][5], HmnR_np[ir * num_wann ** 2 + n * num_wann + m][6])
                irvec[ir][0] = HmnR_np[ir * num_wann ** 2][0]
                irvec[ir][1] = HmnR_np[ir * num_wann ** 2][1]
                irvec[ir][2] = HmnR_np[ir * num_wann ** 2][2]
            
            self.irvec = irvec
            self.ndegen = ndegen
            self.HmnR_np_iR = HmnR_np_iR

        return num_wann, ndegen, irvec, HmnR_np_iR

    def is_hermitian(self, matrix):
        return np.allclose(matrix, np.conj(matrix.T), atol=1e-4, )

    def assembly_ham(self,klist, irvec, ndegen, lat, lattice_moduli, HmnR_np_iR):
        for i in range(len(klist)):
            Ham_bulk = np.zeros((self.nwan, self.nwan), dtype=complex)
            for iR in range(self.nrpts):
                ia = irvec[iR][0]
                ib = irvec[iR][1]
                ic = irvec[iR][2]
                R = ia * lat[0, :] + ib * lat[1, :] + ic * lat[2, :]
                kdotR = np.dot(R, klist[i])
                factor = np.exp(1j * kdotR * 2 * np.pi / lattice_moduli[0])
                Ham_bulk[:, :] = Ham_bulk[:, :] + (HmnR_np_iR[:, :, iR] * factor / ndegen[iR])
        self.Ham_bulk = Ham_bulk[:, :]
        print('Ham_bulk is OK')
        return Ham_bulk

    def get_ham_k(self,k):
        Ham_bulk_k = np.zeros((self.nwan, self.nwan), dtype=complex)
        for iR in range(self.nrpts):
            ia = self.irvec[iR][0]
            ib = self.irvec[iR][1]
            ic = self.irvec[iR][2]
            R = ia * self.lattice_vectors[0, :] + ib * self.lattice_vectors[1, :] + ic * self.lattice_vectors[2, :]
            kdotR = np.dot(R, k)
            factor = np.exp(1j * kdotR * 2 * np.pi / self.lattice_moduli[0])
            Ham_bulk_k[:, :] = Ham_bulk_k[:, :] + (self.HmnR_np_iR[:, :, iR] * factor / self.ndegen[iR])
        self.Ham_bulk_k = Ham_bulk_k[:, :]
        print("k is",k)
        print('Ham_bulk is OK',self.Ham_bulk_k)
        return Ham_bulk_k
    
    def sovle_k(self):
        if self.Ham_bulk_k is None:
            raise ValueError("Ham_bulk_k is empty!!")
        eigen_value, eigen_vector = np.linalg.eig(self.Ham_bulk_k)


#    def solve_all_k(self,kpts):


    def get_bands_from_ham(self, kpath, irvec, ndegen, lat, lattice_moduli, HmnR_np_iR):
        band_structure = np.zeros((len(kpath), self.nwan))
        for i in range(len(kpath)):
            Ham_bulk = np.zeros((self.nwan, self.nwan), dtype=complex)
            for iR in range(self.nrpts):
                ia = irvec[iR][0]
                ib = irvec[iR][1]
                ic = irvec[iR][2]
                R = ia * lat[0, :] + ib * lat[1, :] + ic * lat[2, :]
                kdotR = np.dot(R, kpath[i])
                factor = np.exp(1j * kdotR * 2 * np.pi / lattice_moduli[0])
                Ham_bulk[:, :] = Ham_bulk[:, :] + (HmnR_np_iR[:, :, iR] * factor / ndegen[iR])
            matrix_to_check = Ham_bulk[:, :]
            if self.is_hermitian(matrix_to_check):
                pass
            else:
                pass
            eigen_value, eigen_vector = np.linalg.eig(Ham_bulk)
            sorted_eig = np.sort(np.real(eigen_value))
            band_structure[i][:] = sorted_eig
        return band_structure

    def plot_bands(self, kdist, band_wanbandtb, band_structure):
        #plt.figure(figsize=(10, 12))
        #plt.ylim(-12, 4)
        #for i in range(self.nwan):
        #    plt.plot(kdist, band_wanbandtb[:, i], '-', color='red', linewidth=4)
        #    plt.plot(kdist, band_structure[:, i], 'o', color='blue', markersize=2)
        full_path = self.file_path + 'fplo_wann_band.dat'
        full_path2 = self.file_path + 'fplo_wann_band_assemble.dat'
        # with open(full_path, 'w', encoding='utf-8') as f:
        #     for j in len(kdist):
        #         f.write(
        #             f"{line[0]:5d} {line[1]:5d} {line[2]:5d} {int(line[3]):5d} {int(line[4]):5d}\n"
        #         )
        # band_wanbandtb.reshape(len(kdist),self.nwan)
        
        with open(full_path, "w") as file:
            for col in range(self.nwan):
                for row in range(len(kdist)):
                    file.write(f"{row} {band_wanbandtb[row, col]} \n")
                file.write("\n")
        with open(full_path2, "w") as file2:
            for col in range(self.nwan):
                for row in range(len(kdist)):
                    file2.write(f"{row} {band_structure[row, col]} \n")
                file2.write("\n")
        
        #np.savetxt(full_path, band_wanbandtb, fmt="%f")
        #plt.savefig(self.file_path + 'biband.png')
        #plt.show()
        
        
        
    def pauli_block_all(self):
        if self.Ham_bulk is None:
            raise ValueError("Ham_bulk is not initialized. Call assembly_ham first.")
        self.MI = (self.Ham_bulk[::2, ::2] + self.Ham_bulk[1::2, 1::2]) / 2.0
        self.Mx = (self.Ham_bulk[::2, 1::2] + self.Ham_bulk[1::2, ::2]) / 2.0
        # Note that this is not element wise product with sigma_y but dot product
        self.My = (self.Ham_bulk[::2, 1::2] - self.Ham_bulk[1::2, ::2]) * 0.5j
        self.Mz = (self.Ham_bulk[::2, ::2] - self.Ham_bulk[1::2, 1::2]) / 2.0
        return self.MI, self.Mx, self.My, self.Mz
    
    
