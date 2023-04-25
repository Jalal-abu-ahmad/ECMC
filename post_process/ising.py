import os
import random
import sys
import time

from EventChainActions import *
from SnapShot import *
from graph import Graph

epsilon = 1e-8
day = 86400  # sec


class Ising(Graph):
    def __init__(self, sim_path, k_nearest_neighbors, directed=False, centers=None, spheres_ind=None, J=None):
        super().__init__(sim_path, k_nearest_neighbors=k_nearest_neighbors, directed=directed, centers=centers,
                         spheres_ind=spheres_ind, vec_name="ground_state", correlation_name="Cv_vs_J")
        if os.path.exists(self.op_dir_path):
            pattern = re.compile(self.correlation_name + "_.*.txt")
            reals = []
            for file_name in os.listdir(self.op_dir_path):
                if pattern.match(file_name):
                    reals.append(int(re.split('[_|\.]', file_name)[-2]))
            if len(reals) > 0:
                maxreal = None
                maxmatlen = 0
                for r in reals:
                    mat = np.loadtxt(os.path.join(self.op_dir_path, self.correlation_name + "_" + str(r) + ".txt"))
                    if mat.shape[0] >= maxmatlen:
                        maxreal = r
                        maxmatlen = mat.shape[0]
                self.update_centers(np.loadtxt(os.path.join(self.sim_path, str(maxreal))), maxreal)
        self.z_spins = [(1 if p[2] > self.l_z / 2 else -1) for p in self.spheres]
        self.J = J

    def update_centers(self, centers, spheres_ind):
        super().update_centers(centers, spheres_ind)
        self.z_spins = [(1 if p[2] > self.l_z / 2 else -1) for p in self.spheres]

    @property
    def op_name(self):
        return "Ising_" + self.direc_str

    @property
    def anneal_path(self):
        return os.path.join(self.op_dir_path, "anneal_" + str(self.spheres_ind) + '.txt')

    def initialize(self, random_initialization=True, J=None):
        if J is not None:
            self.J = J
        self.op_vec = [((2 * random.randint(0, 1) - 1) if random_initialization else z) for z in self.z_spins]
        self.calc_EM()

    def calc_EM(self):
        self.E = 0
        for i in range(self.N):
            for j in self.nearest_neighbors[i]:
                if j > i:  # dont double count
                    continue
                self.E -= self.J * self.op_vec[i] * self.op_vec[j]
        self.M = 0  # Magnetization is if op_vec, ising spins, is corr or anti corr to up down initial partition
        for s, z in zip(self.op_vec, self.z_spins):
            self.M += s * z

    def Metropolis_flip(self):
        i = random.randint(0, self.N - 1)
        de = 0.0
        for j in self.nearest_neighbors[i]:
            de += 2 * self.J * self.op_vec[i] * self.op_vec[j]
        A = min(1, np.exp(-de))
        u = random.random()
        if u <= A:
            self.op_vec[i] *= -1

    def anneal(self, iterations, dJditer=None, diter_save=1):
        J, E, M = [], [], []

        for i in range(int(iterations / diter_save)):
            if dJditer is None:
                for j in range(diter_save):
                    self.Metropolis_flip()
            else:
                for j in range(diter_save):
                    self.Metropolis_flip()
                    self.J += dJditer(self.J)
            self.calc_EM()
            M.append(self.M)
            E.append(self.E)
            J.append(self.J)
        return J, E, M

    def heat_capacity(self, iterations, diter_save=1):
        """
        Cv = 1/k_B*T^2 * Var(E) in statistical physics notation.
        In our notation, we work with beta*E as E, and so Cv=k_B*Var(E). Working in units of k_B and per particale we 4
        get Cv=Var(E)/N
        """
        _, E, _ = self.anneal(iterations, diter_save=diter_save)
        return np.var(E, ddof=1) / self.N, np.mean(E)

    def frustrated_bonds(self, E, J):
        return 1 / 2 * (1 - np.array(E) / (self.bonds_num * np.array(J)))

    def real_path(self, real):
        return os.path.join(self.op_dir_path, "real_" + str(real) + "_" + str(self.spheres_ind) + '.txt')

    def calc_order_parameter(self, J_range=(-0.4, -3), iterations=None, realizations=100, samples=1000,
                             random_initialization=True, save_annealing=True, localy_freeze=True):
        if iterations is None:
            iterations = self.N * int(1e5)
        diter_save = int(iterations / samples)
        # dJditer = lambda J: 1 / 2 * (-J) ** 3 / (iterations - 1) * (1 / J_range[1] ** 2 - 1 / J_range[0] ** 2)
        # dJditer = lambda J: -2 / iterations * (np.sqrt(-J_range[1]) - np.sqrt(-J_range[0])) * np.sqrt(-J)
        # dJditer = lambda J: (J_range[1] - J_range[0]) / iterations
        dJditer = lambda J: J ** 2 * (1 / J_range[0] - 1 / J_range[1]) / iterations
        self.minE = float('inf')
        self.minEconfig = None
        self.frustration, self.Ms = [], []

        def append_real(J, E, M):
            self.frustration.append(self.frustrated_bonds(E, J))
            self.Ms.append(np.array(M) / self.N)
            self.mE = min(self.frustration[-1])
            if self.mE < self.minE:
                self.minE = self.mE
                self.minEconfig = copy.deepcopy(self.op_vec)
            return

        calculated_reals = 0
        if OrderParameter.exists(self.anneal_path):
            anneal_mat = np.loadtxt(self.anneal_path)
            calculated_reals = int((anneal_mat.shape[1] - 1) / 2)
            if calculated_reals >= realizations:
                return
            for i in range(1, calculated_reals + 1):
                J = anneal_mat[:, 0]
                append_real(J, J * self.bonds_num * (1 - 2 * anneal_mat[:, i]),
                            self.N * anneal_mat[:, calculated_reals + i])
        if OrderParameter.exists(self.vec_path):
            self.minEconfig = np.loadtxt(self.vec_path)
        for i in range(calculated_reals, realizations):
            self.initialize(random_initialization=random_initialization, J=J_range[0])
            J, E, M = self.anneal(iterations, diter_save=diter_save, dJditer=dJditer)
            append_real(J, E, M)
            np.savetxt(self.anneal_path, np.transpose([J] + self.frustration + self.Ms))
            np.savetxt(self.vec_path, self.minEconfig)
        self.op_vec = self.minEconfig
        self.J = -100
        _, _, _ = self.anneal(self.N, diter_save=self.N)
        return

    def correlation(self, Jarr=None, initial_iterations=None, cv_iterations=None, post_refinement=False):
        if initial_iterations is None:
            initial_iterations = int(2e3 * self.N)
        if cv_iterations is None:
            cv_iterations = int(1e4 * self.N)
        if Jarr is None:
            Jarr = [J for J in np.round(np.linspace(-0.05, -1.5, 30), 2)]
            for J in np.round(np.linspace(-0.4, -0.7, 31), 2):
                Jarr.append(J)
            Jarr = np.flip(np.unique(Jarr))
        frustration = []
        Cv = []
        Jarr_calculated = []
        if os.path.exists(self.corr_path):
            Jarr_calculated_np, Cv_np, frustration_np = np.loadtxt(self.corr_path, unpack=True, usecols=(0, 1, 2))
            if type(Jarr_calculated_np) is np.ndarray:
                Jarr_calculated = [J for J in Jarr_calculated_np]
                Cv = [c for c in Cv_np]
                frustration = [f for f in frustration_np]
            else:
                Jarr_calculated = [Jarr_calculated_np]
                Cv = [Cv_np]
                frustration = [frustration_np]
        last_cv_spins_path = os.path.join(self.op_dir_path, "last_cv_spins_" + str(self.spheres_ind))
        if os.path.exists(last_cv_spins_path):
            self.op_vec = np.loadtxt(last_cv_spins_path)
        else:
            self.initialize(J=Jarr[0])
        initial_time = time.time()
        for J in Jarr:
            if J in Jarr_calculated:
                continue
            self.J = J
            _, _, _ = self.anneal(initial_iterations, diter_save=initial_iterations)
            Cv_, E_ = self.heat_capacity(cv_iterations, diter_save=self.N)
            Cv.append(Cv_)
            frustration.append(self.frustrated_bonds(E_, J))
            Jarr_calculated.append(J)
            np.savetxt(self.corr_path, np.transpose([Jarr_calculated, Cv, frustration]))
            np.savetxt(last_cv_spins_path, self.op_vec)
            if time.time() - initial_time > 2 * day:
                sys.exit(7)
        if post_refinement:
            # Refinement
            I = np.argsort(Jarr_calculated)
            J_sorted = np.array(Jarr_calculated)[I]
            Cv_sorted = np.array(Cv)[I]
            imax = np.argmax(Cv_sorted)
            refined_Jarr = np.linspace(J_sorted[imax - 2], J_sorted[imax + 2], 10)
            for J in refined_Jarr:
                if J in Jarr_calculated:
                    continue
                self.J = J
                _, _, _ = self.anneal(initial_iterations, diter_save=initial_iterations)
                Cv_, E_ = self.heat_capacity(cv_iterations, diter_save=self.N)
                Cv.append(Cv_)
                frustration.append(self.frustrated_bonds(E_, J))
                Jarr_calculated.append(J)
                np.savetxt(self.corr_path, np.transpose([Jarr_calculated, Cv, frustration]))
                np.savetxt(last_cv_spins_path, self.op_vec)
        if os.path.exists(last_cv_spins_path):
            os.remove(last_cv_spins_path)
        I = np.argsort(Jarr_calculated)
        self.corr_centers = np.array(Jarr_calculated)[I]
        self.counts = np.array(frustration)[I]
        self.op_corr = np.array(Cv)[I]

    def read_or_calc_write(self, realizations=20, **calc_order_parameter_args):
        if OrderParameter.exists(self.anneal_path):
            anneal_mat = np.loadtxt(self.anneal_path)
            calculated_reals = int((anneal_mat.shape[1] - 1) / 2)
            if calculated_reals >= realizations:
                return
            self.calc_order_parameter(realizations=realizations, **calc_order_parameter_args)
            self.write(write_correlations=False, write_vec=True)
        else:
            super().read_or_calc_write(realizations=realizations, **calc_order_parameter_args)
