from EventChainActions import *
from graph import Graph
from psi_mn import PsiMN

epsilon = 1e-8
day = 86400  # sec


class LocalOrientation(Graph):

    def __init__(self, sim_path, m, n, radius, centers=None, spheres_ind=None, psi_mn=None):
        # directed=True saves computation time, and it does not matter because by symmetry of the metric use radius
        # nearest neighbor graph is already symmetric, that is undirected graph by construction
        if psi_mn is None:
            self.psi_mn = PsiMN(sim_path, m, n, centers=centers, spheres_ind=spheres_ind)
            self.psi_mn.read_or_calc_write()
        else:
            self.psi_mn = psi_mn
        super().__init__(sim_path, radius=radius, directed=True, centers=centers, spheres_ind=spheres_ind,
                         correlation_name="hist_rad=" + str(radius), vec_name="local-psi_rad=" + str(radius))

    @property
    def op_name(self):
        return "Local_" + self.psi_mn.op_name

    def calc_order_parameter(self):
        self.op_vec = copy.deepcopy(self.psi_mn.op_vec)
        for i in range(self.N):
            self.op_vec[i] = np.mean(
                [self.psi_mn.op_vec[j] for j in self.nearest_neighbors[i]] + [self.psi_mn.op_vec[i]])

    def correlation(self):
        self.counts, bin_edges = np.histogram(np.abs(self.op_vec), bins=30)

        self.corr_centers = [1 / 2 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
        self.op_corr = self.counts / np.trapz(self.counts, self.corr_centers)
