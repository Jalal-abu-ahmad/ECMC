import os

import scipy.sparse
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

from EventChainActions import *
from order_parameter import OrderParameter

epsilon = 1e-8
day = 86400  # sec


class Graph(OrderParameter):
    def __init__(self, sim_path, k_nearest_neighbors=None, radius=None, directed=False, centers=None, spheres_ind=None,
                 calc_upper_lower=False, **kwargs):
        self.k = k_nearest_neighbors
        self.radius = radius
        assert (k_nearest_neighbors is not None) or (
                radius is not None), "Graph needs either k nearest neighbors or radius"
        self.directed = directed
        self.graph_father_path = os.path.join(sim_path, "OP", "Graph")

        single_layer_k = 4 if k_nearest_neighbors == 4 else 6
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower, k_nearest_neighbors=single_layer_k, **kwargs)
        # update centers overrides has calc_graph in it so it will be called on super().__init__()
        # extra argument k_nearest_neighbors goes to upper and lower layers

    @property
    def direc_str(self):
        type = "k=" + str(self.k) if (self.k is not None) else "radius=" + str(self.radius)
        direc = "directed" if self.directed else "undirected"
        return type + "_" + direc

    @property
    def graph_file_path(self):
        return os.path.join(self.graph_father_path, self.direc_str + "_" + str(self.spheres_ind) + ".npz")

    @property
    def frustration_path(self):
        return os.path.join(self.graph_father_path,
                            "frustration_" + self.direc_str + "_" + str(self.spheres_ind) + ".txt")

    def calc_graph(self):
        if not os.path.exists(self.graph_father_path):
            os.mkdir(self.graph_father_path)
        recalc_graph = True
        if os.path.exists(self.graph_file_path):
            recalc_graph = False
            self.graph = scipy.sparse.load_npz(self.graph_file_path)
            if self.graph.shape != (self.N, self.N):
                recalc_graph = True
        if recalc_graph:
            def cyc_dist(p1, p2, boundaries):
                dx = np.array(p1) - p2  # direct vector
                dsq = 0
                for i in range(2):
                    L = boundaries[i]
                    dsq += min(dx[i] ** 2, (dx[i] + L) ** 2, (dx[i] - L) ** 2)  # find shorter path through B.D.
                return np.sqrt(dsq)

            cyc = lambda p1, p2: cyc_dist(p1, p2, [self.l_x, self.l_y])
            X = [p[:2] for p in self.spheres]
            if self.k is not None:
                self.graph = kneighbors_graph(X, n_neighbors=self.k, metric=cyc)
            else:
                self.graph = radius_neighbors_graph(X, radius=self.radius, metric=cyc)
            if not self.directed:
                I, J, _ = scipy.sparse.find(self.graph)[:]
                Ed = [(i, j) for (i, j) in zip(I, J)]
                Eud = []
                udgraph = scipy.sparse.csr_matrix((self.N, self.N))
                for i, j in Ed:
                    if ((j, i) in Ed) and ((i, j) not in Eud) and ((j, i) not in Eud):
                        Eud.append((i, j))
                        udgraph[i, j] = 1
                        udgraph[j, i] = 1
                self.graph = udgraph
            scipy.sparse.save_npz(self.graph_file_path, self.graph)
        self.nearest_neighbors = [[j for j in self.graph.getrow(i).indices] for i in range(self.N)]
        self.bonds_num = 0
        for i in range(self.N):
            self.bonds_num += len(self.nearest_neighbors[i])
        self.bonds_num /= 2
        if (not os.path.exists(self.frustration_path)) and (self.radius is None) and (not self.directed):
            frustration = 0
            z_spins = [(1 if p[2] > self.l_z / 2 else -1) for p in self.spheres]
            for i in range(len(self.spheres)):
                for j in self.nearest_neighbors[i]:
                    if j > i:
                        continue
                    if z_spins[i] * z_spins[j] == 1:
                        frustration += 1
            frustration /= self.bonds_num
            np.savetxt(self.frustration_path, [frustration])

    def update_centers(self, centers, spheres_ind):
        super().update_centers(centers, spheres_ind)
        self.calc_graph()
