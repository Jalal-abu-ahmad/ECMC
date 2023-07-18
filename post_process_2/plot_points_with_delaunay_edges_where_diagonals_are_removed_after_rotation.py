import numpy as np
from matplotlib import pyplot as plt

import utils
from post_process_2 import burger_field_calculation, Burger_field_optimization


def demo():

    L = 100
    N = 0

    points = [[0, 0]]
    square_points = [[0, 0]]
    X = np.linspace(0, 500, 501)
    Y = np.linspace(0, 500, 501)
    for i in range(len(X)):
        print(i)
        for j in range(len(X)):
            points = np.append(points, [[X[i], Y[j]]], axis=0)
    points = np.delete(points, 0, 0)
    points = utils.rotate_points_by_angle(points, 0.5, L, L)

    for p in points:
        if 200 <= p[0] <= 300 and 200 <= p[1] <= 300:
            square_points = np.append(square_points, [p-[200, 200]], axis=0)

    N = len(square_points)-1
    square_points = np.delete(square_points, 0, 0)
    plt.scatter(square_points[:, 0], square_points[:, 1])
    plt.show()

    global_theta, b = calculate_rotation_angel_averaging_on_all_sites(points=square_points, l_x=L, l_y=L, N=N)
    print("theta=", global_theta)
    aligned_points = align_points(square_points, L, L, N, square_points, global_theta)
    burger_vecs, list_of_edges = burger_field_calculation.Burger_field_calculation(points=aligned_points, l_x=L, l_y=L,
                                                                                   N=N, global_theta=0, a=1, order=1)

    utils.plot(points=aligned_points, edges_with_colors=list_of_edges, burger_vecs=burger_vecs, non_diagonal=False)

    plt.scatter(aligned_points[:, 0], aligned_points[:, 1])
    plt.gca().set_aspect('equal')
    plt.show()


def calculate_rotation_angel_averaging_on_all_sites(points, l_x, l_y, N):
    print("calculating nearest neighbors graph... will take a while...")
    NNgraph = utils.nearest_neighbors_graph(points=points, l_x=l_x, l_y=l_y, n_neighbors=4)
    psimn_vec = []
    lattice_constant = []
    nearest_neighbors = utils.nearest_neighbors(N=N, NNgraph=NNgraph)
    for i in range(N):
        if (i % 1000) == 0:
            print("angle calculation progress = ", int((i / N) * 100), "%")
        dr = [utils.cyclic_vec([l_x, l_y], points[i], points[j]) for j in nearest_neighbors[i]]
        for r in dr:
            lattice_constant.append(utils.vector_length(r))
        t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
        psi_n = np.mean(np.exp(1j * 4 * t))
        psimn_vec.append(np.abs(psi_n) * np.exp(1j * np.angle(psi_n)))
    psi_avg = np.mean(psimn_vec)
    orientation = np.imag(np.log(psi_avg)) / 4
    a = np.mean(lattice_constant)
    return orientation, a


def align_points(points, l_x, l_y, N, burger_vecs, theta):

    aligned_points = utils.rotate_points_by_angle(points, theta, l_x, l_y)

    # burger(aligned_points)
    # temp1 = utils.rotate_points_by_angle(burger_vecs[:,[0,1]],theta)
    # temp2 = utils.rotate_points_by_angle(burger_vecs[:,[2,3]], theta)
    # rotated_Burger_vec= np.hstack((temp1, temp2))

    return aligned_points


def read_from_file():

    mac = False

    if mac:
        file_path = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.81_AF_square_ECMC/94363239"
        # file_path = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.8_AF_square_ECMC/92549977"

    else:
        file_path = "C:/Users/Galal/ECMC/N=90000_h=0.8_rhoH=0.81_AF_square_ECMC/94363239"
        # file_path = "C:/Users/Galal/ECMC/N=90000_h=0.8_rhoH=0.8_AF_square_ECMC/92549977"

    N = 90000
    rho_H = 0.81
    h = 0.8
    L, a, l_z = utils.get_params(N=N, h=h, rho_H=rho_H)
    # a = L / (np.sqrt(N) - 1)

    points_with_z = utils.read_points_from_file(file_path=file_path)
    points = np.delete(points_with_z, 2, axis=1)
    assert points.shape == (N, 2)
    print("imported data and parameters")
    global_theta, b = calculate_rotation_angel_averaging_on_all_sites(points=points, l_x=L, l_y=L, N=N)

    wrapped_points_with_z = utils.wrap_boundaries(points_with_z, [L, L], int(L/50))
    wrapped_points = np.delete(wrapped_points_with_z, 2, axis=1)
    wrapped_points_z = wrapped_points_with_z[:, 2]

    aligned_points = align_points(wrapped_points, L, L, N, points, global_theta)
    print("rotated points")
    print("theta=", global_theta)
    aligned_points_with_z = np.column_stack((aligned_points, wrapped_points_z))

    Burger_vecs, list_of_edges, is_point_in_dislocation = burger_field_calculation.Burger_field_calculation(points=aligned_points, a=a, order=1)
    print("no of total edges:", len(list_of_edges))
    Burger_field = np.column_stack([Burger_vecs, np.full(len(Burger_vecs), -1)])
    Burger_field_optimization.Burger_vec_pairing(points, list_of_edges, Burger_field, a)

    utils.plot_boundaries([L, L], -global_theta)
    utils.plot(points=aligned_points, edges_with_colors=list_of_edges, burger_vecs=Burger_vecs, non_diagonal=True)
    utils.plot_frustrations(list_of_edges, aligned_points_with_z, aligned_points, l_z, L)
    utils.plot_colored_points(aligned_points_with_z, l_z, is_point_in_dislocation)


if __name__ == "__main__":
    #demo()
    read_from_file()
