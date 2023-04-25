from scipy.optimize import fmin

from EventChainActions import *
from order_parameter import OrderParameter

epsilon = 1e-8
day = 86400  # sec


class BraggStructure(OrderParameter):
    def __init__(self, sim_path, m, n, centers=None, spheres_ind=None):
        self.op_name = "Bragg_S"
        self.k_peak = None
        self.data = []
        self.m, self.n = m, n
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower=False)

    def calc_eikr(self, k):
        return np.exp([1j * (k[0] * r[0] + k[1] * r[1]) for r in self.spheres])

    def S(self, k):
        sum_r = np.sum(self.calc_eikr(k))
        N = len(self.spheres)
        S_ = np.real(1 / N * sum_r * np.conjugate(sum_r))
        self.data.append([k[0], k[1], S_])
        return S_

    def tour_on_circle(self, k_radii, theta=None):
        if theta is None:
            theta_peak = np.arctan2(self.k_peak[1], self.k_peak[0])
            theta = np.mod(theta_peak + np.linspace(0, 1, 101) * 2 * np.pi, 2 * np.pi)
            theta = np.sort(np.concatenate([theta, [np.pi / 4 * x for x in range(8)]]))
        for t in theta:
            self.S(k_radii * np.array([np.cos(t), np.sin(t)]))

    def k_perf(self):
        # rotate self.spheres by orientation before so peak is at for square [1, 0]
        l = np.sqrt(self.l_x * self.l_y / len(self.spheres))
        if self.m == 1 and self.n == 4:
            return 2 * np.pi / l * np.array([1, 0])
        if self.m == 1 and self.n == 6:
            a = np.sqrt(2.0 / np.sqrt(3)) * l
            return 2 * np.pi / a * np.array([1.0, -1.0 / np.sqrt(3)])
        if self.m == 2 and self.n == 3:
            a = np.sqrt(4.0 / np.sqrt(3)) * l
            return 2 * np.pi / a * np.array([0, 2])
        raise NotImplementedError

    def calc_peak(self):
        S = lambda k: -self.S(k)
        # TODO: use rotated k_perf according to sample orientation. This might be the reason I could not get honeycomb
        #  ic's positional correlation to be algebraic (I got always exponential), because I did not find a proper
        #  Bragg peak
        self.k_peak, S_peak_m, _, _, _ = fmin(S, self.k_perf(), xtol=0.01 / len(self.spheres), ftol=1.0,
                                              full_output=True)
        self.S_peak = -S_peak_m

    def peaks_other_angles(self):
        mn = self.m * self.n
        theta0 = np.arctan2(self.k_perf()[1], self.k_perf()[0])  # k_perf is overwritten in magnetic bragg
        return [theta0 + n * 2 * np.pi / mn for n in range(1, mn)]

    def calc_other_peaks(self):
        S = lambda k: -self.S(k)
        self.other_peaks_k = [self.k_peak]
        self.other_peaks_S = [self.S_peak]
        k_radii = np.linalg.norm(self.k_perf())  # k_perf is overwritten in magnetic bragg
        for theta in self.peaks_other_angles():
            k_perf = k_radii * np.array([np.cos(theta), np.sin(theta)])
            k_peak, S_peak_m, _, _, _ = fmin(S, k_perf, xtol=0.01 / len(self.spheres), ftol=1.0, full_output=True)
            self.other_peaks_k.append(k_peak)
            self.other_peaks_S.append(S_peak_m)

    def write(self, write_vec=True, write_correlations=True):
        op_vec = self.op_vec
        self.op_vec = np.array(self.data)
        # overrides the usless e^(ikr) vector for writing the important data in self.data. self.op_corr has already been
        # calculated.
        super().write(write_correlations=write_correlations, write_vec=write_vec, write_upper_lower=False)
        self.op_vec = op_vec

    def calc_order_parameter(self, psi=None):
        if psi is None:
            psi = PsiMN(self.sim_path, self.m, self.n, centers=self.spheres, spheres_ind=self.spheres_ind)
            psi.read_or_calc_write()
        _, self.spheres = psi.rotate_spheres()
        self.calc_peak()
        self.op_vec = self.calc_eikr(self.k_peak)
        k1, k2 = np.linalg.norm(self.k_perf()), np.linalg.norm(self.k_peak)
        m1, m2 = min([k1, k2]), max([k1, k2])
        dm = m2 - m1
        for k_radii in np.linspace(m1 - 10 * dm, m2 + 10 * dm, 12):
            self.tour_on_circle(k_radii)
        self.calc_other_peaks()

    def correlation(self, bin_width=0.1, low_memory=True, randomize=False, realizations=int(1e7), time_limit=2 * day):
        super().correlation(bin_width, False, low_memory, randomize, realizations, time_limit)

    def read_vec(self):
        self.data = np.loadtxt(self.vec_path, dtype=complex)
        S = [d[2] for d in self.data]
        i = np.argmax(S)
        self.S_peak = S[i]
        self.k_peak = [self.data[i][0], self.data[i][1]]
        self.op_vec = self.calc_eikr(self.k_peak)
