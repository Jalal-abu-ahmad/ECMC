from scipy.spatial import Delaunay

from EventChainActions import *
from bragg_structure import BraggStructure
from local_orientation import LocalOrientation
from order_parameter import OrderParameter
from psi_mn import PsiMN

epsilon = 1e-8
day = 86400  # sec


class BurgerField(OrderParameter):

    def __init__(self, sim_path, centers=None, spheres_ind=None, calc_upper_lower=False, orientation_rad=None):
        self.orientation_rad = orientation_rad
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower=False)
        if calc_upper_lower:
            upper_centers = [c for c in self.spheres if c[2] >= self.l_z / 2]
            lower_centers = [c for c in self.spheres if c[2] < self.l_z / 2]
            self.upper = BurgerField(sim_path, centers=upper_centers, spheres_ind=self.spheres_ind)
            self.lower = BurgerField(sim_path, centers=lower_centers, spheres_ind=self.spheres_ind)
            self.upper.op_name = "upper_" + self.op_name
            self.lower.op_name = "lower_" + self.op_name

    @property
    def op_name(self):
        return "burger_vectors" + (
            "" if (self.orientation_rad is None) else ("_orientation_rad=" + str(self.orientation_rad)))

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
        else:
            if self.orientation_rad > 0:
                local_psi_mn = LocalOrientation(self.sim_path, 1, 4, self.orientation_rad, self.spheres,
                                                self.spheres_ind, psi)
                local_psi_mn.read_or_calc_write()
            else:
                local_psi_mn = psi
            orientation_array = np.array([np.imag(np.log(p)) / 4 for p in local_psi_mn.op_vec])
        disloc_burger, disloc_location = BurgerField.calc_burger_vector(self.spheres, [self.l_x, self.l_y],
                                                                        perfect_lattice_vectors, single_orientation,
                                                                        orientation_array)
        self.op_vec = np.concatenate((np.array(disloc_location).T, np.array(disloc_burger).T)).T  # x, y, bx, by field
        if calc_upper_lower:
            self.lower.calc_order_parameter()
            self.upper.calc_order_parameter()

    @staticmethod
    def calc_burger_vector(spheres, boundaries, perfect_lattice_vectors, single_orientation=None,
                           orientation_array=None):
        """
        Calculate the burger vector on each plaquette of the Delaunay triangulation using methods in:
            [1]	https://link.springer.com/content/pdf/10.1007%2F978-3-319-42913-7_20-1.pdf
            [2]	https://www.sciencedirect.com/science/article/pii/S0022509614001331?via%3Dihub
        :param perfect_lattice_vectors: list of vectors of the perfect lattice. Their magnitude is not important.
        :return: The positions (r) and burger vector at each position b. The position of a dislocation is take as the
                center of the plaquette.
        """
        # TODO: orientation should not be by average of theta because of modulu, it should be average of psi's and then
        #  Imag of log
        # TODO: add warning for final snapping of Burgers so it would not be too large.
        wraped_centers, orientation_array = BurgerField.wrap_with_boundaries(spheres, boundaries, w=5,
                                                                             orientation_array=orientation_array)
        # all spheres within w distance from cyclic boundary will be mirrored
        tri = Delaunay(wraped_centers)
        dislocation_burger = []
        dislocation_location = []
        for i, simplex in enumerate(tri.simplices):
            rc = np.mean(tri.points[simplex], 0)
            if not ((0 < rc[0] < boundaries[0]) and (0 < rc[1] < boundaries[1])):
                continue
            b_i = None
            if orientation_array is not None:
                b_i = BurgerField.burger_calculation(wraped_centers[simplex], perfect_lattice_vectors,
                                                     nodes_orientation=orientation_array[simplex])
            else:
                R = BurgerField.rotation_matrix(single_orientation)
                lattice_vectors = [np.matmul(R, p) for p in perfect_lattice_vectors]
                b_i = BurgerField.burger_calculation(wraped_centers[simplex], lattice_vectors)
            if np.linalg.norm(b_i) > epsilon:
                dislocation_location.append(rc)
                dislocation_burger.append(b_i)
        return dislocation_burger, dislocation_location

    @staticmethod
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    @staticmethod
    def burger_calculation(simplex_points, global_reference_lattice, nodes_orientation=None):
        simplex_points = np.array(simplex_points)

        ts = []
        rc = np.mean(simplex_points, 0)
        for p in simplex_points:
            ts.append(np.arctan2(p[0] - rc[0], p[1] - rc[1]))
        I = np.argsort(ts)
        simplex_points = simplex_points[I]  # calculate burger circuit always anti-clockwise
        if nodes_orientation is not None:
            nodes_orientation = np.array(nodes_orientation)[I]
        Ls = []
        reference_lattice = global_reference_lattice
        for (a, b) in [(0, 1), (1, 2), (2, 0)]:
            if nodes_orientation is not None:
                theta_edge = (nodes_orientation[a] + nodes_orientation[b]) / 2
                R_edge = BurgerField.rotation_matrix(theta_edge)
                reference_lattice = [np.matmul(R_edge, p) for p in global_reference_lattice]
            Ls.append(BurgerField.L_ab(simplex_points[b] - simplex_points[a], reference_lattice))
        if nodes_orientation is not None:
            theta_simplex = np.mean(nodes_orientation)
            R_simplex = BurgerField.rotation_matrix(theta_simplex)
            reference_lattice = [np.matmul(R_simplex, p) for p in global_reference_lattice]
            return BurgerField.L_ab(np.sum(Ls, 0), reference_lattice)
        return np.sum(Ls, 0)

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
