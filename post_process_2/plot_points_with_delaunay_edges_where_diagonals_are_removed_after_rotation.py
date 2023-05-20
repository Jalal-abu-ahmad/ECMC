from scipy.spatial import Delaunay
import matplotlib as plt
import utils
import numpy as np

from Structure import Metric


def demo():
    # Generate a square lattice of points
    L = 20  # Size of the square lattice
    N = 100  # Number of points
    noise = 0.1
    rotation_angle_degree = 90
    rotation_angle = np.deg2rad(rotation_angle_degree)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), np.sin(rotation_angle)],
        [-np.sin(rotation_angle), np.cos(rotation_angle)],
    ])
    x = np.linspace(0, L, int(np.sqrt(N)))
    y = np.linspace(0, L, int(np.sqrt(N)))
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).T
    rotated_points = points @ rotation_matrix
    assert rotated_points.shape == (N, 2)
    # print(points)

    # Add some random defects to the lattice
    noise = np.random.normal(0, noise, size=rotated_points.shape)
    rotated_points += noise
    aligned_points = align_points(rotated_points)


    utils.plot_points_with_delaunay_edges_where_diagonals_are_removed(points=aligned_points,L=L,N=N)

    utils.plot_points_with_delaunay_edges_where_diagonals_are_removed(points=rotated_points,L=L,N=N)


def calculate_rotation_angle(points ,a):
    tri = Delaunay(points)
    list_of_angles = []
    edges = utils.delaunay2edges(tri)
    utils.filter_diagonal_edges(array_of_edges=edges, a=a, points=points)
    x_hat = np.array([1, 0])
    for e in edges:
        vec = points[e[1]] - points[e[0]]
        if utils.is_in_pos_x_direction(e, points):
            list_of_angles.append(utils.calculate_angle_between_two_vectors(vec, x_hat))
    theta = np.mean(list_of_angles)

    return theta


def calculate_rotation_angel_averaging_on_all_sites(points, l_x, l_y, N):
    NNgraph = utils.nearest_neighbors_graph(points=points, l_x=l_x, l_y=l_y, n_neighbors=4)
    psimn_vec = []
    nearest_neighbors = utils.nearest_neighbors(N=N, NNgraph=NNgraph)
    for i in range(len(points)):
        dr = [Metric.cyclic_vec([l_x, l_y], points[i], points[j]) for j in
              nearest_neighbors[i]]
        t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
        psi_n = np.mean(np.exp(1j * n * t))
        psimn_vec[i] = np.abs(psi_n) * np.exp(1j * self.m * np.angle(psi_n))
    return psimn_vec


def align_points(points, a):
    theta = calculate_rotation_angle(points=points, a=a)
    aligned_points = utils.rotate_points_by_angle(points,theta)
    return aligned_points


def read_from_file():
    file_path = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.82_AF_triangle_ECMC/84426366"
    N = 90000
    rho_H = 0.82
    h=0.8
    L = utils.get_L(N=N, h=h, rho_H=rho_H)
    a = L / (np.sqrt(N) - 1)
    points = utils.read_points_from_file(file_path=file_path)
    assert points.shape == (N, 2)
    aligned_points=align_points(points=points, a=a)
    utils.plot_points_with_delaunay_edges_where_diagonals_are_removed(points=aligned_points, N=N, L=L)



if __name__ == "__main__":
    read_from_file()
