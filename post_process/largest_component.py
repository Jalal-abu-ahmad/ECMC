import scipy.sparse

from EventChainActions import *
from magnetic_topological_corr import MagneticTopologicalCorr

epsilon = 1e-8
day = 86400  # sec


class LargestComponent(MagneticTopologicalCorr):
    def __init__(self, sim_path, k_nearest_neighbors, directed=False, centers=None, spheres_ind=None):
        super().__init__(sim_path, k_nearest_neighbors=k_nearest_neighbors, directed=False, centers=None,
                         spheres_ind=None, correlation_name="largest_component")

    def correlation(self, calc_upper_lower=False):
        I, J, _ = scipy.sparse.find(self.graph)[:]
        E = [(i, j) for (i, j) in zip(I, J)]
        self.graph = scipy.sparse.csr_matrix((self.N, self.N))
        for i, j in E:
            if self.op_vec[i] * self.op_vec[j] < 0:
                self.graph[i, j] = 1
                self.graph[j, i] = 1
        n_components, labels = scipy.sparse.csgraph.connected_components(self.graph, directed=False)
        largest_component = 0
        for l in np.unique(labels):
            component_size = [l for l in labels].count(l)
            if component_size > largest_component:
                largest_component = component_size
        self.counts = 0
        self.op_corr = largest_component / self.N
        self.corr_centers = 0
