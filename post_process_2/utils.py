import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.neighbors import kneighbors_graph


def less_first(a, b):
    return [a, b] if a < b else [b, a]


def get_closest_vector_in_length(original_vec, list_of_vecs) -> float:
    minimum_dist = float('inf')
    for v in list_of_vecs:
        diff = np.linalg.norm(v - original_vec)
        minimum_dist = min(minimum_dist, diff)
    return minimum_dist


def calculate_angle_between_two_vectors(v1, v2):
    return np.arccos(dot_product(v1, v2)/(vector_length(v1) * vector_length(v2)))


def vector_length(v):
    return np.sqrt(dot_product(v, v))


def dot_product(v1, v2):
    product = sum((a * b) for a, b in zip(v1, v2))  # general n-dimesnsions vector
    return product


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotate_points_by_angle(points, angle):
    rotated_points = points @ rotation_matrix(angle)
    return rotated_points


def cyc_position_alignment(points, boundaries):
    for p in points:
        if p[0] > boundaries[0]:
            p[0] = p[0] - boundaries[0]

        if p[0] < 0:
            p[0] = p[0] + boundaries[0]

        if p[1] > boundaries[1]:
             p[1] = p[1] - boundaries[1]

        if p[1] < 0:
             p[1] = p[1] + boundaries[1]


def cyc_dist(p1, p2, boundaries):
    dx = np.array(p1) - p2  # direct vector
    dsq = 0
    for i in range(2):
        L = boundaries[i]
        dsq += min(dx[i] ** 2, (dx[i] + L) ** 2, (dx[i] - L) ** 2)  # find shorter path through B.D.
    return np.sqrt(dsq)


def is_horizontal(edge, points):
    x = np.array([[1, 0], [-1, 0]])
    y = np.array([[0, 1], [0, -1]])
    vec = edge[1] - edge[0]
    x_diff = get_closest_vector_in_length(vec, x)
    y_diff = get_closest_vector_in_length(vec, y)
    if x_diff < y_diff:
        return True
    return False


def is_diagonal(edge, perfect_lattice_vectors_only_diags, perfect_lattice_vectors_only_no_diags, points) -> bool:
    vec = edge[1] - edge[0]
    x = get_closest_vector_in_length(vec, perfect_lattice_vectors_only_diags)
    y = get_closest_vector_in_length(vec, perfect_lattice_vectors_only_no_diags)

    if x > y:
        return False
    else:
        return True


def delaunay2edges(tri):
    list_of_edges = []
    iterations = len(tri.simplices)
    for i, triangle in enumerate(tri.simplices):
        if (i % 1000) == 0:
            print("delaunay2edges progress = ", int((i / iterations) * 100), "%")
        for e1, e2 in [[0, 1], [1, 2], [2, 0]]:  # for all edges of triangle
            edge = less_first(triangle[e1], triangle[e2])  # always lesser index first
            list_of_edges.append(edge)
    array_of_edges = np.unique(list_of_edges, axis=0)  # remove duplicates
    return array_of_edges


def NN2edges(points, nearest_neighbours, L):
    list_of_edges = []
    for p in range(len(nearest_neighbours)):
        nn_edges = []
        for n in range(len(nearest_neighbours[p])):
            edge = (points[p], points[nearest_neighbours[p][n]])
            if vec_length(edge[0],edge[1]) < L/2:
                nn_edges.append(edge)
        list_of_edges.append(nn_edges)
    return list_of_edges


def vec_length(p1,p2):
    return math.dist(p1,p2)


def nearest_neighbors_graph(points, l_x, l_y, n_neighbors):
    cyc = lambda p1, p2: cyc_dist(p1, p2, [l_x, l_y])
    NNgraph = kneighbors_graph(points, n_neighbors=n_neighbors, metric=cyc)
    return NNgraph


def nearest_neighbors(N, NNgraph):
    return [[j for j in NNgraph.getrow(i).indices] for i in range(N)]


def get_lengths_of_edges(tri, array_of_edges):
    list_of_lengths = []
    for p1, p2 in array_of_edges:
        x1, y1 = tri.points[p1]
        x2, y2 = tri.points[p2]
        list_of_lengths.append((x1 - x2) ** 2 + (y1 - y2) ** 2)
    array_of_lengths = np.sqrt(np.array(list_of_lengths))
    return array_of_lengths


def cyclic_vec(boundaries, sphere1, sphere2):
    """
    :return: vector pointing from sphere2 to sphere1 ("sphere1-sphere2") shortest through cyclic boundaries
    """
    dx = np.array(sphere1) - sphere2  # direct vector
    vec = np.zeros(len(dx))
    # vec[2] = dx[2]
    for i in range(2):
        l = boundaries[i]
        dxs = np.array([dx[i], dx[i] + l, dx[i] - l])
        vec[i] = dxs[np.argmin(dxs ** 2)]  # find shorter path through B.D.
    return vec


def filter_diagonal_edges(array_of_edges, a, points, rotation_angel, order):
    # calculate a vectors
    a1, a2 = np.array([a, 0]), np.array([0, a])

    # get lattice vectors
    perfect_lattice_vectors_only_diags = filter_none(
        [(n * a1 + m * a2 if n != 0 and m != 0 else None) for n in range(-order, order+1) for m in range(-order, order+1)]
    )
    perfect_lattice_vectors_only_no_diags = filter_none(
        [(n * a1 + m * a2 if (n == 0 or m == 0) and not (n == 0 and m == 0) else None) for n in range(-order, order+1) for m in
         range(-order, order+1)]
    )

    perfect_lattice_vectors_only_no_diags_aligned = rotate_points_by_angle(perfect_lattice_vectors_only_no_diags, rotation_angel)
    perfect_lattice_vectors_only_diags_aligned = rotate_points_by_angle(perfect_lattice_vectors_only_diags, rotation_angel)

    list_of_edges = []
    iterations = len(array_of_edges)
    for i, edge in enumerate(array_of_edges):
        if (i % 1000) == 0:
            print("filter_edges progress = ", int((i / iterations) * 100), "%")
        if not is_diagonal(edge, perfect_lattice_vectors_only_diags_aligned, perfect_lattice_vectors_only_no_diags_aligned, points):
            list_of_edges.append(edge)
    array_of_edges = np.unique(list_of_edges, axis=0)  # remove duplicates
    return array_of_edges


def filter_none(l: list) -> list:
    return list(filter(lambda item: item is not None, l))


def plot_colored_points(points, l_z):
    print("coloring the graph")
    for p in points:
        if p[2] > l_z/2:
            plt.plot(p[0], p[1], 'ro',markersize=5)
        else:
            plt.plot(p[0], p[1], 'bo',markersize=5)

    plt.axis([160, 230, 340, 390])
    plt.gca().set_aspect('equal')
    plt.show()


def plot_frustrations(array_of_edges, points_with_z, l_z):
    print("coloring frustrations green")
    for (p1, p2) in array_of_edges:
        if (points_with_z[p1][2] > l_z/2 and points_with_z[p2][2] > l_z/2) or (points_with_z[p1][2] < l_z/2 and points_with_z[p2][2] < l_z/2):
            x1 = points_with_z[p1][0]
            y1 = points_with_z[p1][1]
            x2 = points_with_z[p2][0]
            y2 = points_with_z[p2][1]
            plt.plot([x1, x2], [y1, y2], color='green')


def plot(points, edges_with_colors, burger_vecs, non_diagonal):
    print("plotting edges")
    for (p1, p2), color in edges_with_colors:
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        if not(color == 'red' and non_diagonal==True):
            plt.plot([x1, x2], [y1, y2], color=color, alpha=1)

    # tri = Delaunay(points)
    # plt.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices, color='red', alpha=0.7)

    print("plotting Burger field")
    if burger_vecs is not None:
        plt.quiver(burger_vecs[:, 0], burger_vecs[:, 1], burger_vecs[:, 2], burger_vecs[:, 3])


def plot_points_with_no_edges(points):
    plot(points=points, edges_with_colors=[])


def plot_points_with_delaunay_edges_where_diagonals_are_removed(points_with_z, alignment_angel, burger_vecs,a,l_z):
    # a = L / (np.sqrt(N) - 1)

    points = np.delete(points_with_z, 2, axis=1)

    # Perform Delaunay triangulation and get edges
    tri = Delaunay(points)
    edges = delaunay2edges(tri)

    # remove edges that are diagonal
    array_of_edges = filter_diagonal_edges(array_of_edges=edges, a=a, points=points, rotation_angel=alignment_angel, order=1)

    # edges
    edges_with_colors = []
    for e in array_of_edges:
        if is_horizontal(e, points):
            edges_with_colors.append((e, "y"))
        else:
            edges_with_colors.append((e, "blue"))

    plot(points=points, edges_with_colors=edges_with_colors, burger_vecs=burger_vecs)

    plot_frustrations(array_of_edges, points_with_z,l_z)


def plot_nn_graph(nn_edges, points):

    for n in range(len(nn_edges)):
        for e in range(len(nn_edges[n])):
            x1, y1 = nn_edges[n][e][0]
            x2, y2 = nn_edges[n][e][1]
            plt.plot([x1, x2], [y1, y2],color='blue')
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


def get_params(N, h, rho_H):
    # n_row = int(np.sqrt(N))
    # n_col = n_row  # Square initial condition for n_row!=n_col is not implemented...
    r, sig = 1.0, 2.0
    A = N * sig ** 2 / (rho_H * (1 + h))
    a = np.sqrt(A / N)
    n_row_cells, n_col_cells = int(np.sqrt(A) / (a * np.sqrt(2))), int(np.sqrt(A) / (a * np.sqrt(2)))
    edge = np.sqrt(A / (n_row_cells * n_col_cells))
    l_x = edge * n_col_cells
    l_y = edge * n_row_cells
    assert abs(l_x - l_y) < 0.000000001
    l_z = (h + 1) * sig
    return l_x, a, l_z


def perfect_lattice_vectors(a, order):
    # a = L / (np.sqrt(N) - 1)

    a1, a2 = np.array([a, 0]), np.array([0, a])

    # get lattice vectors
    perfect_lattice_vectors_only_diags = filter_none(
        [(n * a1 + m * a2 if n != 0 and m != 0 else None) for n in range(-order, order+1) for m in range(-order, order+1)]
    )
    perfect_lattice_vectors_only_no_diags = filter_none(
        [(n * a1 + m * a2 if (n == 0 or m == 0) and not (n == 0 and m == 0) else None) for n in range(-order, order+1) for m in
         range(-order, order+1)]
    )

    return perfect_lattice_vectors_only_diags, perfect_lattice_vectors_only_no_diags


def read_points_from_file(file_path: str) -> np.ndarray:

    # load points from file
    points = np.loadtxt(file_path)

    return points