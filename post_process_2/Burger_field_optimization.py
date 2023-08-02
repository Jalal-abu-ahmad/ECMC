import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
from shapely.geometry import Point, Polygon
from scipy.optimize import linear_sum_assignment

from post_process_2 import utils

epsilon = 0.00001


def midpoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]


def Burger_vec_pairing(points, list_of_edges, Burger_field, a, boundaries):

    Burger_field_diagonals_separated = break_diagonal_vecs_2_components(Burger_field)
    up_vecs, down_vecs, right_vecs, left_vecs = vecs_classification(Burger_field_diagonals_separated)

    up = calculate_vectors_midpoints(up_vecs)
    down = calculate_vectors_midpoints(down_vecs)
    right = calculate_vectors_midpoints(right_vecs)
    left = calculate_vectors_midpoints(left_vecs)

    pair_vecs(up, down, right, left, boundaries)

    utils.plot_burger_field(Burger_field)
    plt.gca().set_aspect('equal')
    plt.show()


def break_diagonal_vecs_2_components(vector_field):

    separated_diagonals_vector_field = []
    for [p1_x, p1_y, p2_x, p2_y], neighbor in vector_field:
        if is_vec_diagonal([p1_x, p1_y, p2_x, p2_y]):
            separated_diagonals_vector_field.append(break_diagonal_vec([p1_x, p1_y, p2_x, p2_y])[0])
            separated_diagonals_vector_field.append(break_diagonal_vec([p1_x, p1_y, p2_x, p2_y])[1])
        else:
            separated_diagonals_vector_field.append([p1_x, p1_y, p2_x, p2_y])

    return separated_diagonals_vector_field


def is_vec_diagonal(vec):
    if vec[0] != vec[2]:
        if vec[1] != vec[3]:
            return True
    return False


def break_diagonal_vec(vec):
    horizontal_vec = [vec[0], vec[1], vec[2], vec[1]]
    vertical_vec = [vec[0], vec[1], vec[0], vec[3]]

    return horizontal_vec, vertical_vec


def calculate_vectors_midpoints(vector_field):
    mid_vec = []
    for [p1_x, p1_y, p2_x, p2_y] in vector_field:
        mid_vec.append(midpoint([p1_x, p1_y], [p2_x, p2_y]))

    return mid_vec


def vecs_classification(vector_field):
    up_vecs = []
    down_vecs = []
    left_vecs = []
    right_vecs = []

    for [p1_x, p1_y, p2_x, p2_y] in vector_field:
        if p1_x == p2_x:
            if p1_y < p2_y:
                up_vecs.append([p1_x, p1_y, p2_x, p2_y])
            else:
                down_vecs.append([p1_x, p1_y, p2_x, p2_y])
        if p1_y == p2_y:
            if p1_x < p2_x:
                right_vecs.append([p1_x, p1_y, p2_x, p2_y])
            else:
                left_vecs.append([p1_x, p1_y, p2_x, p2_y])

    return up_vecs, down_vecs, right_vecs, left_vecs


def pair_vecs(up_vecs, down_vecs, right_vecs, left_vecs, boundaries):

    up_down_cost_matrix = calculate_cost_matrix(up_vecs, down_vecs, boundaries)
    right_left_cost_matrix = calculate_cost_matrix(right_vecs, left_vecs, boundaries)

    up_down_row_ind, up_down_col_ind = linear_sum_assignment(up_down_cost_matrix)
    right_left_row_ind, right_left_col_ind = linear_sum_assignment(right_left_cost_matrix)

    connect_and_plot_pairs(up_vecs, down_vecs, up_down_col_ind, up_down_row_ind)
    connect_and_plot_pairs(right_vecs, left_vecs, right_left_col_ind, right_left_row_ind)


def calculate_cost_matrix(first_side, second_side, boundaries):

    cost_matrix = np.zeros((len(first_side), len(second_side)))

    for i in range(len(first_side)):
        for j in range(len(second_side)):
            cost_matrix[i][j] = np.exp(utils.cyc_dist(first_side[i], second_side[j], boundaries))

    return cost_matrix


def connect_and_plot_pairs(first_side, second_side, col_ind, row_ind):

    for i in range(len(row_ind)):
        if utils.vec_length_from_2_points([first_side[row_ind[i]][0],first_side[row_ind[i]][1]] , [second_side[col_ind[i]][0], second_side[col_ind[i]][1]]) < 300:
            plt.plot([first_side[row_ind[i]][0], second_side[col_ind[i]][0]], [first_side[row_ind[i]][1], second_side[col_ind[i]][1]], color="purple")


"""------------------------------------------------------------------------------------------------------------------"""


def Burger_vec_pairing_old(points, list_of_edges, Burger_field, a):
    Burger_vecs_centers = []
    for [p1_x, p1_y, p2_x, p2_y], neighbor in Burger_field:
        Burger_vecs_centers.append(midpoint([p1_x, p1_y], [p2_x, p2_y]))
    iterated_pairing(Burger_field, Burger_vecs_centers, no_of_iterations=20, no_of_neighbors=4)
    isolate_unsatisfied_paths(Burger_field, Burger_vecs_centers, list_of_edges, points, a)


def iterated_pairing(Burger_field, Burger_vecs_centers, no_of_iterations, no_of_neighbors):

    unpaired_vecs = []
    for i in range(len(Burger_vecs_centers)):
        unpaired_vecs.append([Burger_vecs_centers[i], int(i)])
    for iteration in range(no_of_iterations):
        print("iteration no:", iteration)
        vec_centers_only = [unpaired_vecs[i][0] for i in range(len(unpaired_vecs))]
        NNgraph = kneighbors_graph(vec_centers_only, n_neighbors=no_of_neighbors)
        check_and_arrange_pairing(NNgraph, Burger_field, unpaired_vecs)
        unpaired_vecs = clean_up(Burger_field, unpaired_vecs)


def check_and_arrange_pairing(NNgraph, Burger_field, unpaired_vecs):
    # check if:
    # 1) the closest neighbors for a pair (if A is NN of b then B is NN of A)
    # 2) check if the pairs cancel out
    # -> pair up the vectors that satisfy 1) & 2)
    nearest_neighbors = utils.nearest_neighbors(len(unpaired_vecs), NNgraph)

    for i in range(len(nearest_neighbors)):
        for neighbor in nearest_neighbors[i]:
            if i_is_neighbor(i, nearest_neighbors, neighbor):
                if add_up_to_zero(Burger_field[unpaired_vecs[i][1]], Burger_field[unpaired_vecs[neighbor][1]]):
                    if Burger_field[unpaired_vecs[i][1]][1] == -1 and Burger_field[unpaired_vecs[neighbor][1]][1] == -1:
                        Burger_field[unpaired_vecs[i][1]][1] = unpaired_vecs[neighbor][1]
                        Burger_field[unpaired_vecs[neighbor][1]][1] = unpaired_vecs[i][1]


def i_is_neighbor(i, nearest_neighbors, neighbor):
    for n in nearest_neighbors[neighbor]:
        if n == i:
            return True
    return False


def clean_up(Burgers_field, last_iteration_of_pairing):
    # remove approved pairs and keep unpaired vecs to start next iteration

    unpaired_vecs = []
    for [p_x, p_y], original_index in last_iteration_of_pairing:
        if Burgers_field[original_index][1] == -1:
            unpaired_vecs.append([[p_x, p_y], original_index])

    return unpaired_vecs


def isolate_unsatisfied_paths(Burger_field, Burger_vecs_centers, list_of_edges, points, a):
    # isolate all bonds and points between pairs as they are in a non-neutral zone
    # use : "https://www.matecdev.com/posts/point-in-polygon.html"
    # "https://shapely.readthedocs.io/en/latest/manual.html#binary-predicates"

    create_list_of_polygons(a, Burger_field)


def add_up_to_zero(vec1, vec2):
    sum = utils.two_points_sum(utils.two_points_2_vector(vec1[0][0:2], vec1[0][2:4]),
                               utils.two_points_2_vector(vec2[0][0:2], vec2[0][2:4]))
    for i in sum:
        if np.abs(i) > epsilon:
            return False

    return True


def create_list_of_polygons(a, Burger_field):

    list_of_polygons = []
    buffer = 1.2

    for [p1_x, p1_y, p2_x, p2_y], neighbor in Burger_field:
        if neighbor != -1:
            p1, p2 = less_first(Burger_field[neighbor][0], [p1_x, p1_y, p2_x, p2_y])
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="purple")
            p11 = p1 + np.array([-buffer*a, buffer*a])
            p12 = p1 + np.array([-buffer*a, -buffer*a])

            p21 = p2 + np.array([buffer*a, buffer*a])
            p22 = p2 + np.array([buffer*a, -buffer*a])

            poly = Polygon([p11, p21, p22, p12])
            list_of_polygons.append(poly)

            x, y = poly.exterior.xy
            plt.plot(x, y, color='#6699cc', alpha=0.8,
                        linewidth=1.5, solid_capstyle='round', zorder=2)

    return list_of_polygons


def less_first(vec1, vec2):
    p1 = np.array(midpoint(vec1[0:2], vec1[2:4]))
    p2 = np.array(midpoint(vec2[0:2], vec2[2:4]))

    if p1[0] > p2[0]:
        return p2, p1
    return p1, p2
