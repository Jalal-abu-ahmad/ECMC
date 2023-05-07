import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def less_first(a, b):
    return [a, b] if a < b else [b, a]


def get_closest_vector_in_length(original_vec, list_of_vecs) -> float:
    minimum_dist = float('inf')
    for v in list_of_vecs:
        diff = np.linalg.norm(v - original_vec)
        minimum_dist = min(minimum_dist, diff)
    return minimum_dist


def calculate_angle_between_two_vectors(v1, v2):
    return np.arccos(dot_product(v1,v2)/(vector_length(v1) * vector_length(v2)))


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

def is_in_pos_x_direction(edge, points):
    x=np.array([1,0])
    y=np.array([0,1])
    vec = points[edge[1]] - points[edge[0]]
    x_diff = np.linalg.norm(x - vec)
    y_diff = np.linalg.norm(y - vec)
    if x_diff < y_diff:
        return True
    return False


def is_diagonal(edge, perfect_lattice_vectors_only_diags, perfect_lattice_vectors_only_no_diags, points) -> bool:
    vec = points[edge[1]] - points[edge[0]]
    x = get_closest_vector_in_length(vec, perfect_lattice_vectors_only_diags)
    y = get_closest_vector_in_length(vec, perfect_lattice_vectors_only_no_diags)
    # print("x = ", x, ", y = ",  y, ", edge = ", edge, ", vec = ", vec)
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


def get_lengths_of_edges(tri, array_of_edges):
    list_of_lengths = []
    for p1, p2 in array_of_edges:
        x1, y1 = tri.points[p1]
        x2, y2 = tri.points[p2]
        list_of_lengths.append((x1 - x2) ** 2 + (y1 - y2) ** 2)
    array_of_lengths = np.sqrt(np.array(list_of_lengths))
    return array_of_lengths


def filter_diagonal_edges(array_of_edges, a, points):
    # calculate a vectors
    a1, a2 = np.array([a, 0]), np.array([0, a])

    # get lattice vectors
    perfect_lattice_vectors_only_diags = filter_none(
        [(n * a1 + m * a2 if n != 0 and m != 0 else None) for n in range(-3, 4) for m in range(-3, 4)]
    )
    perfect_lattice_vectors_only_no_diags = filter_none(
        [(n * a1 + m * a2 if (n == 0 or m == 0) and not (n == 0 and m == 0) else None) for n in range(-3, 4) for m in
         range(-3, 4)]
    )

    # perfect_lattice_vectors = [n * a1 + m * a2 for n in range(-3, 3) for m in range(-3, 3)]
    # print("perfect_lattice_vectors_only_diags = ", perfect_lattice_vectors_only_diags)
    # print("perfect_lattice_vectors_only_no_diags = ", perfect_lattice_vectors_only_no_diags)

    list_of_edges = []
    iterations = len(array_of_edges)
    for i, edge in enumerate(array_of_edges):
        if (i % 1000) == 0:
            print("filter_edges progress = ", int((i / iterations) * 100), "%")
        if not is_diagonal(edge, perfect_lattice_vectors_only_diags, perfect_lattice_vectors_only_no_diags, points):
            list_of_edges.append(edge)
    array_of_edges = np.unique(list_of_edges, axis=0)  # remove duplicates
    return array_of_edges


def filter_none(l: list) -> list:
    return list(filter(lambda item: item is not None, l))


def plot(points, edges_with_colors):
    for (p1, p2), color in edges_with_colors:
        x1, y1 = points[p1]
        x2, y2 = points[p2]
        plt.plot([x1, x2], [y1, y2], color=color)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


def plot_points_with_no_edges(points):
    plot(points=points, edges_with_colors=[])


def plot_points_with_delaunay_edges_where_diagonals_are_removed(points, L, N):
    a = L / (np.sqrt(N) - 1)

    # Perform Delaunay triangulation and get edges
    tri = Delaunay(points)
    edges = delaunay2edges(tri)

    # remove edges that are diagonal
    array_of_edges = filter_diagonal_edges(array_of_edges=edges, a=a, points=points)

    # edges
    edges_with_colors = []
    for e in array_of_edges:
        edges_with_colors.append((e, "blue"))
    plot(points=points, edges_with_colors=edges_with_colors)


def plot_points_with_delaunay_edges(points, L, N):
    # Perform Delaunay triangulation
    tri = Delaunay(points)

    a = L / (np.sqrt(N) - 1)
    a1, a2 = np.array([a, 0]), np.array([0, a])
    # perfect_lattice_vectors = [n * a1 + m * a2 for n in range(-3, 3) for m in range(-3, 3)]
    perfect_lattice_vectors_only_diags = filter_none(
        [(n * a1 + m * a2 if n != 0 and m != 0 else None) for n in range(-3, 4) for m in range(-3, 4)]
    )
    # print("perfect_lattice_vectors_only_diags = ", perfect_lattice_vectors_only_diags)
    perfect_lattice_vectors_only_no_diags = filter_none(
        [(n * a1 + m * a2 if (n == 0 or m == 0) and not (n == 0 and m == 0) else None) for n in range(-3, 4) for m in range(-3, 4)]
    )

    # print("perfect_lattice_vectors_only_no_diags = ", perfect_lattice_vectors_only_no_diags)

    edges = delaunay2edges(tri)

    # print(len(edges))

    array_of_edges = np.unique(edges, axis=0)  # remove duplicates
    plt.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices, color='red')
    x_lines = []
    y_lines = []
    for p1, p2 in array_of_edges:
        x1, y1 = points[p1]
        x2, y2 = points[p2]
        plt.plot([x1, x2], [y1, y2], color='blue')

    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


def get_L(N, h, rho_H):
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
    # l_z = (h + 1) * sig
    return l_x



def read_points_from_file(file_path: str) -> np.ndarray:
    # load points from file
    points = np.loadtxt(file_path)
    # delete z column from points
    points_without_z = np.delete(points, 2, axis=1)
    # return
    return points_without_z

