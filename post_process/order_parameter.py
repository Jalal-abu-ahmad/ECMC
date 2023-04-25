import time

from SnapShot import *

epsilon = 1e-8
day = 86400  # sec


class OrderParameter:

    def __init__(self, sim_path, centers=None, spheres_ind=None, calc_upper_lower=False, vec_name="vec",
                 correlation_name="correlation", **kwargs):
        self.sim_path = sim_path
        self.write_or_load = WriteOrLoad(sim_path)
        self.op_vec = None
        self.op_corr = None
        self.corr_centers = None
        self.counts = None
        self.vec_name = vec_name
        self.correlation_name = correlation_name
        self.op_father_dir = os.path.join(self.sim_path, "OP")
        self.update_centers(centers, spheres_ind)
        if not os.path.exists(self.op_father_dir): os.mkdir(self.op_father_dir)
        if not os.path.exists(self.op_dir_path): os.mkdir(self.op_dir_path)
        if calc_upper_lower:
            upper_centers = [c for c in self.spheres if c[2] >= self.l_z / 2]
            lower_centers = [c for c in self.spheres if c[2] < self.l_z / 2]
            self.upper = type(self)(sim_path, centers=upper_centers, spheres_ind=self.spheres_ind,
                                    calc_upper_lower=False, **kwargs)
            self.lower = type(self)(sim_path, centers=lower_centers, spheres_ind=self.spheres_ind,
                                    calc_upper_lower=False, **kwargs)
            self.upper.op_name = "upper_" + self.op_name
            self.lower.op_name = "lower_" + self.op_name

    @property
    def op_dir_path(self):
        return os.path.join(self.op_father_dir, self.op_name)

    @property
    def mean_vs_real_path(self):
        return os.path.join(self.op_dir_path, 'mean_vs_real.txt')

    @property
    def vec_path(self):
        return os.path.join(self.op_dir_path, self.vec_name + "_" + str(self.spheres_ind) + '.txt')

    @property
    def corr_path(self):
        return os.path.join(self.op_dir_path, self.correlation_name + "_" + str(self.spheres_ind) + '.txt')

    def update_centers(self, centers, spheres_ind):
        if (centers is None) or (spheres_ind is None):
            centers, spheres_ind = self.write_or_load.last_spheres()
        self.spheres_ind = spheres_ind
        self.l_x, self.l_y, self.l_z, rad, rho_H, edge, n_row, n_col = self.write_or_load.load_Input()
        self.spheres = centers
        self.N = len(centers)

    def calc_order_parameter(self, calc_upper_lower=False):
        """to be override by child class"""
        pass

    def correlation(self, bin_width=0.1, calc_upper_lower=False, low_memory=True, randomize=False,
                    realizations=int(1e7), time_limit=2 * day):
        if self.op_vec is None: self.calc_order_parameter()
        lx, ly = self.l_x, self.l_y
        l = np.sqrt(lx ** 2 + ly ** 2) / 2
        centers = np.linspace(0, np.ceil(l / bin_width) * bin_width, int(np.ceil(l / bin_width)) + 1) + bin_width / 2
        kmax = len(centers) - 1
        counts = np.zeros(len(centers))
        phiphi_hist = np.zeros(len(centers))
        init_time = time.time()
        N = len(self.spheres)
        if low_memory:
            if randomize:
                for realization in range(realizations):
                    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
                    phi_phi, k = self.__pair_corr__(i, j, bin_width)
                    k = int(min(k, kmax))
                    counts[k] += 2
                    phiphi_hist[k] += 2 * np.real(phi_phi)
                    if realization % 1000 == 0 and time.time() - init_time > time_limit:
                        break
            else:
                for i in range(N):
                    for j in range(i):  # j<i, j=i not interesting and j>i double counting accounted for in counts
                        phi_phi, k = self.__pair_corr__(i, j, bin_width)
                        k = int(min(k, kmax))
                        counts[k] += 2  # r-r' and r'-r
                        phiphi_hist[k] += 2 * np.real(phi_phi)  # a+a'=2Re(a)
                realization = N * (N - 1) / 2
        else:
            N = len(self.op_vec)
            v = np.array(self.op_vec).reshape(1, N)
            phiphi_vec = (np.conj(v) * v.T).reshape((N ** 2,))
            x = np.array([r[0] for r in self.spheres])
            y = np.array([r[1] for r in self.spheres])
            dx = (x.reshape((len(x), 1)) - x.reshape((1, len(x)))).reshape(len(x) ** 2, )
            dy = (y.reshape((len(y), 1)) - y.reshape((1, len(y)))).reshape(len(y) ** 2, )
            dx = np.minimum(np.abs(dx), np.minimum(np.abs(dx + lx), np.abs(dx - lx)))
            dy = np.minimum(np.abs(dy), np.minimum(np.abs(dy + ly), np.abs(dy - ly)))
            pairs_dr = np.sqrt(dx ** 2 + dy ** 2)

            I = np.argsort(pairs_dr)
            pairs_dr = pairs_dr[I]
            phiphi_vec = phiphi_vec[0, I]
            i = 0
            for j in range(len(pairs_dr)):
                if pairs_dr[j] > centers[i] + bin_width / 2:
                    i += 1
                phiphi_hist[i] += np.real(phiphi_vec[0, j])
                counts[i] += 1
            realization = N ** 2
        print("\nTime Passed: " + str((time.time() - init_time) / day) + " days.\nSummed " + str(
            realization) + " pairs")
        self.counts = counts
        self.op_corr = phiphi_hist / counts
        self.corr_centers = centers

        if calc_upper_lower:
            self.lower.correlation(bin_width, low_memory=low_memory, randomize=randomize, realizations=realizations)
            self.upper.correlation(bin_width, low_memory=low_memory, randomize=randomize, realizations=realizations)

    def __pair_corr__(self, i, j, bin_width):
        lx, ly = self.l_x, self.l_y
        r, r_ = self.spheres[i], self.spheres[j]
        dr_vec = np.array(r) - r_
        dx = np.min(np.abs([dr_vec[0], dr_vec[0] + lx, dr_vec[0] - lx]))
        dy = np.min(np.abs([dr_vec[1], dr_vec[1] + ly, dr_vec[1] - ly]))
        dr = np.sqrt(dx ** 2 + dy ** 2)
        k = int(np.floor(dr / bin_width))
        return self.op_vec[i] * np.conjugate(self.op_vec[j]), k

    def write(self, write_correlations=True, write_vec=False, write_upper_lower=False):
        if write_vec:
            if self.op_vec is None: raise (Exception("Should calculate vec before writing"))
            np.savetxt(self.vec_path, self.op_vec)
        if write_correlations:
            if self.op_corr is None: raise (Exception("Should calculate correlation before writing"))
            np.savetxt(self.corr_path, np.transpose([self.corr_centers, self.op_corr, self.counts]))
        if write_upper_lower:
            self.lower.write(write_correlations, write_vec, write_upper_lower=False)
            self.upper.write(write_correlations, write_vec, write_upper_lower=False)

    @staticmethod
    def exists(file_path):
        if not os.path.exists(file_path): return False
        A = np.loadtxt(file_path, dtype=complex)  # complex most general so this part would not raise error
        if A is None: return False
        if len(A) == 0: return False
        return True

    def read_vec(self):
        self.op_vec = np.loadtxt(self.vec_path, dtype=complex)

    def read_or_calc_write(self, **calc_order_parameter_args):
        if OrderParameter.exists(self.vec_path):
            self.read_vec()
        else:
            self.calc_order_parameter(**calc_order_parameter_args)
            self.write(write_correlations=False, write_vec=True)

    def calc_for_all_realizations(self, calc_mean=True, calc_correlations=True, calc_vec=True, **correlation_kwargs):
        init_time = time.time()
        op_father_dir = os.path.join(self.sim_path, "OP")
        op_dir = os.path.join(op_father_dir, self.op_name)
        if not os.path.exists(op_dir): os.mkdir(op_dir)
        mean_vs_real_reals, mean_vs_real_mean = [], []
        if os.path.exists(self.mean_vs_real_path):
            mat = np.loadtxt(self.mean_vs_real_path, dtype=complex)
            if not mat.shape == (2,):
                mean_vs_real_reals = [int(np.real(r)) for r in mat[:, 0]]
                mean_vs_real_mean = [p for p in mat[:, 1]]
        i = 0
        realizations = self.write_or_load.realizations()
        realizations.append(0)

        while time.time() - init_time < 2 * day and i < len(realizations):
            print(f"Iteration {i} / {len(realizations)}")
            sp_ind = realizations[i]
            if sp_ind != 0:
                centers = np.loadtxt(os.path.join(self.sim_path, str(sp_ind)))
            else:
                centers = np.loadtxt(os.path.join(self.sim_path, 'Initial Conditions'))
            self.update_centers(centers, sp_ind)
            if calc_vec:
                self.read_or_calc_write()
            if calc_mean and (sp_ind not in mean_vs_real_reals):
                mean_vs_real_reals.append(sp_ind)
                mean_vs_real_mean.append(np.mean(self.op_vec))
                I = np.argsort(mean_vs_real_reals)
                sorted_reals = np.array(mean_vs_real_reals)[I]
                sorted_mean = np.array(mean_vs_real_mean)[I]
                np.savetxt(self.mean_vs_real_path, np.array([sorted_reals, sorted_mean]).T)
            if calc_correlations and (not OrderParameter.exists(self.corr_path)):
                self.correlation(**correlation_kwargs)
                self.write(write_correlations=True, write_vec=False)
            i += 1
