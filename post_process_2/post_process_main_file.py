import csv

import numpy as np
import utils
from post_process_2 import burger_field_calculation, Burger_field_optimization, connectivity_Bipartiteness_AFism, \
    testing
import matplotlib.pyplot as plt


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

    aligned_points = utils.rotate_points_by_angle(points, theta)

    # burger(aligned_points)
    # temp1 = utils.rotate_points_by_angle(burger_vecs[:,[0,1]],theta)
    # temp2 = utils.rotate_points_by_angle(burger_vecs[:,[2,3]], theta)
    # rotated_Burger_vec= np.hstack((temp1, temp2))

    return aligned_points


def read_from_file(N, rho_H, h, file_path=None, destination_path=None):

    writer = csv.writer(destination_path, lineterminator='\n')

    # mac = True
    #
    # if mac:
    #     # file_path = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.81_AF_square_ECMC/94363239"
    #     file_path = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.8_AF_square_ECMC/92549977"
    #
    # else:
    #     file_path = "C:/Users/Galal/ECMC/N=90000_h=0.8_rhoH=0.81_AF_square_ECMC/94363239"
    #     # file_path = "C:/Users/Galal/ECMC/N=90000_h=0.8_rhoH=0.8_AF_square_ECMC/92549977"

    # N = 90000
    # rho_H = 0.8
    # h = 0.8
    L, a, l_z = utils.get_params(N=N, h=h, rho_H=rho_H)

    points_with_z = utils.read_points_from_file(file_path=file_path)
    unwrapped_aligned_points_z = points_with_z[:, 2]
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
    unwrapped_aligned_points = utils.rotate_points_by_angle(points, global_theta)
    unwrapped_aligned_points_with_z = np.column_stack((unwrapped_aligned_points, unwrapped_aligned_points_z))

    Burger_vecs, list_of_edges, is_point_in_dislocation = burger_field_calculation.Burger_field_calculation(points=aligned_points, a=a, order=1)
    print("no of total edges:", len(list_of_edges))
    optimized_Burgers_field, pairs_connecting_lines = Burger_field_optimization.Burger_vec_optimization(aligned_points, list_of_edges, Burger_vecs, a, [L, L], global_theta)

    parameters = connectivity_Bipartiteness_AFism.connectivity_Bipartiteness_AFism(list_of_edges, unwrapped_aligned_points_with_z, [L, L], global_theta, l_z)

    # utils.plot_boundaries([L, L], global_theta)
    # utils.plot_burger_field(optimized_Burgers_field, pairs_connecting_lines, [L, L], True)
    # utils.plot(points=aligned_points, edges_with_colors=list_of_edges, non_diagonal=True)
    # utils.plot_frustrations(list_of_edges, aligned_points_with_z, aligned_points, l_z, L)
    # utils.plot_colored_points(aligned_points_with_z, l_z, is_point_in_dislocation)

    return parameters

if __name__ == "__main__":
    read_from_file()
