import os

import scipy.sparse
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.spatial import Delaunay
from EventChainActions import *
from order_parameter import OrderParameter
from bragg_structure import BraggStructure
from psi_mn import PsiMN

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

    def calc_order_parameter(self, calc_upper_lower=False):
        psi = PsiMN(self.sim_path, 1, 4, centers=self.spheres, spheres_ind=self.spheres_ind)
        psi.read_or_calc_write()

        bragg = BraggStructure(self.sim_path, 1, 4, self.spheres, self.spheres_ind)
        bragg.read_or_calc_write(psi=psi)
        a = 2 * np.pi / np.linalg.norm(bragg.k_peak)
        a1, a2 = np.array([a, 0]), np.array([0, a])
        perfect_lattice_vectors = [n * a1 + m * a2 for n in range(-3, 3) for m in range(-3, 3)]
        single_orientation, orientation_array = None, None
        if self.orientation_rad is None:
            single_orientation = psi.rotate_spheres(calc_spheres=False)
            orientation_array = np.array([np.imag(np.log(p)) / 4 for p in local_psi_mn.op_vec])
        Graph.delaunny_triangulation_graph(self.spheres, [self.l_x, self.l_y], perfect_lattice_vectors,
                                           single_orientation, orientation_array)
        self.op_vec = np.concatenate((np.array(disloc_location).T, np.array(disloc_burger).T)).T  # x, y, bx, by field

    def delaunny_triangulation_graph(spheres, boundaries, perfect_lattice_vectors, single_orientation=None,
                           orientation_array=None):
        wraped_centers, orientation_array = Graph.wrap_with_boundaries(spheres, boundaries, w=5,
                                                                             orientation_array=orientation_array)

        tri = Delaunay(wraped_centers)
        for i, simplex in enumerate(tri.simplices):
            rc = np.mean(tri.points[simplex], 0)
            if not ((0 < rc[0] < boundaries[0]) and (0 < rc[1] < boundaries[1])):
                continue
            b_i = None
            if orientation_array is not None:
                b_i = Graph.burger_calculation(wraped_centers[simplex], perfect_lattice_vectors,
                                                     nodes_orientation=orientation_array[simplex])
            else:
                R = Graph.rotation_matrix(single_orientation)
                lattice_vectors = [np.matmul(R, p) for p in perfect_lattice_vectors]
                b_i = Graph.burger_calculation(wraped_centers[simplex], lattice_vectors)
    def L_ab(x_ab, refference_lattice):
        i = np.argmin([np.linalg.norm(x_ab - L) for L in refference_lattice])
        return refference_lattice[i]

    @staticmethod
    def wrap_with_boundaries(spheres, boundaries, w, orientation_array=None):
        centers = np.array(spheres)[:, :2]
        Lx, Ly = boundaries[:2]
        x = centers[:, 0]
        y = centers[:, 1]

        sp1 = centers[np.logical_and(x - Lx > -w, y < w), :] + [-Lx, Ly]
        sp2 = centers[y < w, :] + [0, Ly]
        sp3 = centers[np.logical_and(x < w, y < w), :] + [Lx, Ly]
        sp4 = centers[x - Lx > -w, :] + [-Lx, 0]
        sp5 = centers[:, :] + [0, 0]
        sp6 = centers[x < w, :] + [Lx, 0]
        sp7 = centers[np.logical_and(x - Lx > -w, y - Ly > -w), :] + [-Lx, -Ly]
        sp8 = centers[y - Ly > -w, :] + [0, -Ly]
        sp9 = centers[np.logical_and(x < w, y - Ly > -w), :] + [Lx, -Ly]

        wraped_centers = np.concatenate((sp5, sp1, sp2, sp3, sp4, sp6, sp7, sp8, sp9))
        wraped_orientation = None
        if orientation_array is not None:
            # Copied code from above...
            sp1 = orientation_array[np.logical_and(x - Lx > -w, y < w)]
            sp2 = orientation_array[y < w]
            sp3 = orientation_array[np.logical_and(x < w, y < w)]
            sp4 = orientation_array[x - Lx > -w]
            sp5 = orientation_array[:]
            sp6 = orientation_array[x < w]
            sp7 = orientation_array[np.logical_and(x - Lx > -w, y - Ly > -w)]
            sp8 = orientation_array[y - Ly > -w]
            sp9 = orientation_array[np.logical_and(x < w, y - Ly > -w)]

            wraped_orientation = np.concatenate((sp5, sp1, sp2, sp3, sp4, sp6, sp7, sp8, sp9))
        return wraped_centers, wraped_orientation