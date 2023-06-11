from scipy.spatial import Delaunay
import matplotlib as plt
import utils
import numpy as np

from Structure import Metric


def demo():
    # Generate a square lattice of points
    L = 200  # Size of the square lattice
    N = 10000  # Number of points
    noise = 0
    rotation_angle_degree = 5
    boundaries=([L,L])
    epsilon= L/np.sqrt(N)/2
    rotation_angle = np.deg2rad(rotation_angle_degree)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), np.sin(rotation_angle)],
        [-np.sin(rotation_angle), np.cos(rotation_angle)] ])
    x = np.linspace(epsilon, L-epsilon, int(np.sqrt(N)))
    y = np.linspace(epsilon, L-epsilon, int(np.sqrt(N)))
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).T
    rotated_points = points @ rotation_matrix
    assert rotated_points.shape == (N, 2)

    # Add some random defects to the lattice
    noise = np.random.normal(0, noise, size=rotated_points.shape)
    rotated_points += noise
    # utils.cyc_position_alignment(rotated_points, boundaries)
    aligned_points = align_points(rotated_points, l_x=L, l_y=L, N=N)

    utils.plot_points_with_delaunay_edges_where_diagonals_are_removed(points=aligned_points, L=L, N=N)

    utils.plot_points_with_delaunay_edges_where_diagonals_are_removed(points=rotated_points, L=L, N=N)


def calculate_rotation_angle(points ,a):
    tri = Delaunay(points)
    list_of_angles = []
    edges = utils.delaunay2edges(tri)
    utils.filter_diagonal_edges(array_of_edges=edges, a=a, points=points, order=3)
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
    for i in range(N):
        dr = [utils.cyclic_vec([l_x, l_y], points[i], points[j]) for j in nearest_neighbors[i]]
        t = np.arctan2([np.abs(r[1]) for r in dr], [np.abs(r[0]) for r in dr])
        psi_n = np.mean(np.exp(1j * 4 * t))
        psimn_vec.append(np.abs(psi_n) * np.exp(1j * i * np.angle(psi_n)))
    psi_avg = np.mean(psimn_vec)
    orientation = np.imag(np.log(psi_avg))/4
    return orientation


def align_points(points, l_x, l_y, N,burger_vecs):
    theta = calculate_rotation_angel_averaging_on_all_sites(points=points, l_x=l_x, l_y=l_y , N=N)
    aligned_points = utils.rotate_points_by_angle(points, theta)

    burger(aligned_points)
    #temp1= utils.rotate_points_by_angle(burger_vecs[:,[0,1]],theta)
    #temp2 = utils.rotate_points_by_angle(burger_vecs[:,[2,3]], theta)
    #rotated_Burger_vec= np.hstack((temp1, temp2))
    return aligned_points, rotated_Burger_vec


def read_from_file():

    mac = True

    if mac:
        file_path = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.82_AF_triangle_ECMC/84426366"
        burger_vectors_path="/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.82_AF_triangle_ECMC/OP/burger_vectors/vec_84426366.txt"
    else:
        file_path="C:/Users/Galal/ECMC/N=90000_h=0.8_rhoH=0.81_AF_square_ECMC/94363239"
        burger_vectors_path="C:/Users/Galal/ECMC/N=90000_h=0.8_rhoH=0.81_AF_square_ECMC/OP/burger_vectors/vec_94363239.txt"

    N = 90000
    rho_H = 0.82
    h=0.8
    L = utils.get_L(N=N, h=h, rho_H=rho_H)
    a = L / (np.sqrt(N) - 1)
    points = utils.read_points_from_file(file_path=file_path)
    assert points.shape == (N, 2)
    burger_vecs = np.loadtxt(burger_vectors_path)
    aligned_points, rotated_Burger_vec = align_points(points,L,L,N,burger_vecs)
    #theta = calculate_rotation_angel_averaging_on_all_sites(points=points, l_x=L, l_y=L , N=N)
    utils.plot_points_with_delaunay_edges_where_diagonals_are_removed(points=aligned_points, N=N, L=L, alignment_angel=0, burger_vecs=rotated_Burger_vec)


if __name__ == "__main__":
    #demo()
    read_from_file()
