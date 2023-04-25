import time

from sklearn.utils.graph import single_source_shortest_path_length

from EventChainActions import *
from graph import Graph

epsilon = 1e-8
day = 86400  # sec


class MagneticTopologicalCorr(Graph):
    def __init__(self, sim_path, k_nearest_neighbors, directed=False, centers=None, spheres_ind=None,
                 calc_upper_lower=False, **kwargs):
        super().__init__(sim_path, k_nearest_neighbors=k_nearest_neighbors, directed=directed, centers=centers,
                         spheres_ind=spheres_ind, calc_upper_lower=calc_upper_lower, **kwargs)

    @property
    def op_name(self):
        return "gM_" + self.direc_str

    def calc_order_parameter(self):
        rad, lz = 1.0, self.l_z
        self.op_vec = [(r[2] - lz / 2) / (lz / 2 - rad) for r in self.spheres]

    def correlation(self, calc_upper_lower=False):
        N = len(self.spheres)
        kbound = N  # eassier than finding graph's diameter
        kmax = 0
        counts = np.zeros(kbound + 1)
        phiphi_hist = np.zeros(kbound + 1)
        init_time = time.time()
        for i in range(N):
            shortest_paths_from_i = single_source_shortest_path_length(self.graph, i)
            # Complicated implementation because the simple one returns NxN matrix, and another simple option of node to
            # node shortest path required additional libs and sklearn was already installed for me
            for j in range(N):
                try:
                    k = int(shortest_paths_from_i[j])
                except KeyError:  # j is not in a connected component of i
                    continue
                if k > kmax: kmax = k
                phi_phi = (-1) ** k * self.op_vec[i] * self.op_vec[j]
                counts[k] += 1
                phiphi_hist[k] += phi_phi
        realization = N ** 2
        print("\nTime Passed: " + str((time.time() - init_time) / day) + " days.\nSummed " + str(
            realization) + " pairs")
        counts = counts[:kmax + 1]
        phiphi_hist = phiphi_hist[:kmax + 1]
        self.counts = counts
        self.op_corr = phiphi_hist / counts
        self.corr_centers = np.array(range(kmax + 1))

        if calc_upper_lower:
            self.lower.correlation()
            self.upper.correlation()
