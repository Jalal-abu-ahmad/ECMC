import csv
import os
import re
import sys

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd

import Burger_field_optimization
import burger_field_calculation
import connectivity_Bipartiteness_AFism
import utils
import correlation_functions
import min_distance_Burgers_pairing as mn_dst_pairing

epsilon = 0.00001


def calculate_rotation_angel_averaging_on_all_sites(points, l_x, l_y, N):
    print("calculating nearest neighbors graph... will take a while...")
    NNgraph = utils.nearest_neighbors_graph(points=points, l_x=l_x, l_y=l_y, n_neighbors=4)
    psimn_vec = []
    nearest_neighbors = utils.nearest_neighbors(N=N, NNgraph=NNgraph)
    for i in range(N):
        if (i % 1000) == 0:
            print("angle calculation progress = ", int((i / N) * 100), "%")
        dr = [utils.cyclic_vec([l_x, l_y], points[i], points[j]) for j in nearest_neighbors[i]]
        t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
        psi_n = np.mean(np.exp(1j * 4 * t))
        psimn_vec.append(np.abs(psi_n) * np.exp(1j * np.angle(psi_n)))
    psi_avg = np.mean(psimn_vec)
    orientation = np.imag(np.log(psi_avg)) / 4
    return orientation, NNgraph


def calculate_rotation_angel_of_area(points, l_x, l_y, N):

    NNgraph = utils.nearest_neighbors_graph(points=points, l_x=l_x, l_y=l_y, n_neighbors=4)
    psimn_vec = []
    nearest_neighbors = utils.nearest_neighbors(N=N, NNgraph=NNgraph)
    for i in range(N):
        dr = [utils.no_cyclic_vec([l_x, l_y], points[i], points[j]) for j in nearest_neighbors[i] if utils.no_cyclic_vec([l_x, l_y], points[i], points[j]) is not None]
        if dr:
            t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
            psi_n = np.mean(np.exp(1j * 4 * t))
            psimn_vec.append(np.abs(psi_n) * np.exp(1j * np.angle(psi_n)))
    psi_avg = np.mean(psimn_vec)
    orientation = np.imag(np.log(psi_avg)) / 4
    return orientation


def align_points(points, l_x, l_y, N, burger_vecs, theta):

    aligned_points = utils.rotate_points_by_angle(points, theta)

    return aligned_points


def params_from_name(name):
    ss = re.split("[_=]", name)
    for i, s in enumerate(ss):
        if s == 'N':
            N = int(ss[i + 1])
        if s == 'h':
            h = float(ss[i + 1])
        if s == 'rhoH':
            rhoH = float(ss[i + 1])
        if s == 'triangle' or s == 'square':
            ic = s
            if ss[i - 1] == 'AF' and s == 'triangle':
                ic = 'honeycomb'
    return N, h, rhoH, ic


def Burgers_field_second_iteration(points, a, order, Burger_vecs):
    Burger_vecs_second_itiration = []
    for [p1_x, p1_y, p2_x, p2_y] in Burger_vecs:
        poly = create_polygon(a, [p1_x, p1_y], 5)
        points_in_dislocation_area = np.array(is_point_in_polygon(poly, points))
        lx_sub, ly_sub = shift_sub_lattice_to_origin(points_in_dislocation_area)
        theta_sub = calculate_rotation_angel_of_area(points_in_dislocation_area, lx_sub, ly_sub, len(points_in_dislocation_area))
        aligned_sub_points = utils.rotate_points_by_angle(points_in_dislocation_area, theta_sub)
        Burgers_vecs_in_sub_lattice, edges, points_in = burger_field_calculation.Burger_field_calculation(points=aligned_sub_points, a=a, order=1)
        relocate_vectors(Burgers_vecs_in_sub_lattice, theta_sub, ly_sub, ly_sub)
        for [q1_x, q1_y, q2_x, q2_y] in Burgers_vecs_in_sub_lattice:
            Burger_vecs_second_itiration.append([q1_x, q1_y, q2_x, q2_y])

    return Burger_vecs_second_itiration


def create_polygon(a, point, buffer):

    p11 = point + np.array([-buffer * a, -buffer * a])
    p12 = point + np.array([-buffer * a, buffer * a])
    p21 = point + np.array([buffer * a, buffer * a])
    p22 = point + np.array([buffer * a, -buffer * a])

    poly = Polygon([p11, p12, p21, p22])

    return poly


def is_point_in_polygon(polygon, points):
    no_of_points = len(points)
    point_in_area = []

    for i in range(no_of_points):
        p = Point(points[i])
        if p.within(polygon):
            point_in_area.append(points[i])

    return point_in_area


def shift_sub_lattice_to_origin(points):
    min_x = points[np.argmin(points[:, 0])][0]
    min_y = points[np.argmin(points[:, 1])][1]

    for p in points:
        p[0] = p[0] - min_x
        p[1] = p[1] - min_y

    lx_sub = points[np.argmax(points[:, 0])][0]
    ly_sub = points[np.argmax(points[:, 1])][1]

    return lx_sub, ly_sub


def relocate_vectors(Burgers_vecs, theta_sub, l_x, l_y):
    Burgers_vecs = utils.rotate_Burger_vecs(Burgers_vecs, -theta_sub)
    for i in range(len(Burgers_vecs)):
        Burgers_vecs[i][0] = Burgers_vecs[i][0] + l_x
        Burgers_vecs[i][1] = Burgers_vecs[i][1] + l_y
        Burgers_vecs[i][2] = Burgers_vecs[i][2] + l_x
        Burgers_vecs[i][3] = Burgers_vecs[i][3] + l_y


def post_process_main(sim_name, file_number):


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

    data_prefix = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/"
    result_prefix = "/Users/jalal/Desktop/ECMC"

    # data_prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results3.0/"
    # result_prefix = "/storage/ph_daniel/jalal/ECMC_post_process_results/"

    N, h, rho_H, ic = params_from_name(sim_name)
    L, a, l_z = utils.get_params(N=N, h=h, rho_H=rho_H)
    file_path = os.path.join(data_prefix, sim_name, file_number)
    f = open(os.path.join(result_prefix, 'post_process_results' + sim_name + '.csv'), 'a')
    writer = csv.writer(f, lineterminator='\n')

    points_with_z = utils.read_points_from_file(file_path=file_path)
    unwrapped_aligned_points_z = points_with_z[:, 2]
    points = np.delete(points_with_z, 2, axis=1)
    assert points.shape == (N, 2)
    print("imported data and parameters")
    global_theta, NNgraph = calculate_rotation_angel_averaging_on_all_sites(points=points, l_x=L, l_y=L, N=N)

    wrapped_points_with_z = utils.wrap_boundaries(points_with_z, [L, L], int(L/50))
    wrapped_points = np.delete(wrapped_points_with_z, 2, axis=1)
    wrapped_points_z = wrapped_points_with_z[:, 2]

    aligned_points = align_points(wrapped_points, L, L, N, points, global_theta)
    print("rotated points")
    print("theta=", global_theta)
    # correlation_functions.create_distance_histogram(aligned_points, [L,L] , a, NNgraph)
    aligned_points_with_z = np.column_stack((aligned_points, wrapped_points_z))
    unwrapped_aligned_points = utils.rotate_points_by_angle(points, global_theta)
    unwrapped_aligned_points_with_z = np.column_stack((unwrapped_aligned_points, unwrapped_aligned_points_z))

    Burger_vecs, list_of_edges, is_point_in_dislocation = burger_field_calculation.Burger_field_calculation(points=aligned_points, a=a, order=1)
    print("no of total edges:", len(list_of_edges))

    # proximity_dist = mn_dst_pairing.make_pairs_distances_list(Burger_vecs, L)
    # print(proximity_dist)
    # Burgers_field_second_iteration(aligned_points, a, 1, Burger_vecs)

    utils.plot(points=aligned_points, edges_with_colors=list_of_edges, no_diagonal=True, line_style='--')

    optimized_Burgers_field, pairs_connecting_lines, Burgers_parameters = Burger_field_optimization.Burger_vec_optimization(aligned_points, list_of_edges, Burger_vecs, a, [L, L], global_theta)
    connectivity_parameters, AF_order_parameter = connectivity_Bipartiteness_AFism.connectivity_Bipartiteness_AFism(list_of_edges, unwrapped_aligned_points_with_z, [L, L], global_theta, l_z)
    parameters = list(connectivity_parameters) + list(Burgers_parameters)

    # utils.plot_boundaries([L, L], global_theta)
    utils.plot_burger_field(optimized_Burgers_field, pairs_connecting_lines, [L, L], True)
    utils.plot(points=aligned_points, edges_with_colors=list_of_edges, no_diagonal=True, line_style='solid')
    utils.plot_frustrations(list_of_edges, aligned_points_with_z, aligned_points, l_z, L)
    utils.plot_colored_points(aligned_points_with_z, l_z)

    #plt.axis([130, 200, 360, 410])
    #plt.gca().set_aspect('equal')

    plt.show()

    row = [file_number] + list(parameters)
    writer.writerow(row)


if __name__ == "__main__":

    # sim_name = sys.argv[1]
    # file_number = sys.argv[2]
    # operation = sys,argv[3]
    # post_process_main(sim_name, file_number, operation)

    post_process_main('N=90000_h=0.8_rhoH=0.81_AF_square_ECMC', '90167817')

 # 'N=90000_h=0.8_rhoH=0.81_AF_square_ECMC', '7484757'
 # 'N=90000_h=0.8_rhoH=0.775_AF_square_ECMC', '28391483'