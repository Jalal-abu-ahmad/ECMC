def calc_order_parameter(self, calc_upper_lower=False):
    psi = PsiMN(self.sim_path, 1, 4, centers=self.spheres, spheres_ind=self.spheres_ind)
    psi.read_or_calc_write()

    bragg = BraggStructure(self.sim_path, 1, 4, self.spheres, self.spheres_ind)
    bragg.read_or_calc_write(psi=psi)
    a = 2 * np.pi / np.linalg.norm(bragg.k_peak)
    a1, a2 = np.array([a, 0]), np.array([0, a])
    perfect_lattice_vectors = [n * a1 + m * a2 for n in range(-3, 3) for m in range(-3, 3)]
    single_orientation, orientation_array = None, None
    single_orientation = psi.rotate_spheres(calc_spheres=False)
    Graph.delaunny_triangulation_graph(self.spheres, [self.l_x, self.l_y], perfect_lattice_vectors,
                                       single_orientation, orientation_array)


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
            R_edge = Graph.rotation_matrix(theta_edge)
            reference_lattice = [np.matmul(R_edge, p) for p in global_reference_lattice]
        Ls.append(Graph.L_ab(simplex_points[b] - simplex_points[a], reference_lattice))
    if nodes_orientation is not None:
        theta_simplex = np.mean(nodes_orientation)
        R_simplex = Graph.rotation_matrix(theta_simplex)
        reference_lattice = [np.matmul(R_simplex, p) for p in global_reference_lattice]
        return BurgerField.L_ab(np.sum(Ls, 0), reference_lattice)
    return np.sum(Ls, 0)


@staticmethod
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


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