from EventChainActions import *
from graph import Graph

epsilon = 1e-8
day = 86400  # sec


class PsiMN(Graph):

    def __init__(self, sim_path, m, n, centers=None, spheres_ind=None, calc_upper_lower=False):
        self.m, self.n = m, n
        self.op_name = "psi_" + str(m) + str(n)
        super().__init__(sim_path, k_nearest_neighbors=n, directed=True, centers=centers, spheres_ind=spheres_ind,
                         calc_upper_lower=calc_upper_lower, m=1, n=m * n)
        # extra args m,n goes to upper and lower layers

    def calc_order_parameter(self, calc_upper_lower=False):
        n, centers, graph = self.n, self.spheres, self.graph
        psimn_vec = np.zeros(len(centers), dtype=np.complex)
        cast_sphere = lambda c, r=1, z=0: Sphere([x for x in c] + [z], r)
        for i in range(len(centers)):
            dr = [Metric.cyclic_vec([self.l_x, self.l_y], cast_sphere(centers[i]), cast_sphere(centers[j])) for j in
                  self.nearest_neighbors[i]]
            t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
            psi_n = np.mean(np.exp(1j * n * t))
            psimn_vec[i] = np.abs(psi_n) * np.exp(1j * self.m * np.angle(psi_n))
        self.op_vec = psimn_vec
        if calc_upper_lower:
            self.lower.calc_order_parameter()
            self.upper.calc_order_parameter()

    def rotate_spheres(self, calc_spheres=True):
        n_orientation = self.m * self.n
        psi_avg = np.mean(self.op_vec)
        orientation = np.imag(np.log(psi_avg)) / n_orientation
        if not calc_spheres:
            return orientation
        else:
            R = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0],
                          [0.0, 0.0, 1.0]])  # rotate back from orientation-->0
            return orientation, [np.matmul(R, r) for r in self.spheres]
