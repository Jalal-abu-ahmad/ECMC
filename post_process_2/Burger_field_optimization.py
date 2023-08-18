import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
from shapely.geometry import Point, Polygon, LineString
from scipy.optimize import linear_sum_assignment
import networkx as nx
import geopandas as gpd
from post_process_2 import utils

epsilon = 0.00001


def Burger_vec_optimization(points, list_of_edges, Burger_field, a, boundaries, theta):
    no_duplicates_Burger_field = clean_Burger_field(Burger_field, boundaries, theta)
    Burger_field_diagonals_separated = break_diagonal_vecs_2_components(no_duplicates_Burger_field)
    up_vecs, down_vecs, right_vecs, left_vecs = vecs_classification(Burger_field_diagonals_separated)

    paired_Burgers_field, unpaired_up_down, unpaired_right_left = pair_vecs(up_vecs, down_vecs, right_vecs, left_vecs, boundaries, a)
    second_optimization_pairing(paired_Burgers_field, unpaired_up_down, unpaired_right_left, boundaries, a)
    isolate_edges_that_cross_pairs(paired_Burgers_field, list_of_edges, boundaries, points, a)

    print("no of up vectors=", len(up_vecs))
    print("no of down vectors=", len(down_vecs))
    print("no of right vectors=", len(right_vecs))
    print("no of left vectors=", len(left_vecs))

    return paired_Burgers_field


def clean_Burger_field(Burger_field, boundaries, theta):

    first_rotation = utils.rotate_Burger_vecs(Burger_field, -theta)
    remove_duplicates = utils.remove_points_outside_boundaries(first_rotation, boundaries)
    back_to_normal = utils.rotate_Burger_vecs(remove_duplicates, theta)

    return back_to_normal


def break_diagonal_vecs_2_components(vector_field):

    separated_diagonals_vector_field = []

    for [p1_x, p1_y, p2_x, p2_y] in vector_field:
        if is_vec_diagonal([p1_x, p1_y, p2_x, p2_y]):
            separated_diagonals_vector_field.append(break_diagonal_vec([p1_x, p1_y, p2_x, p2_y])[0])
            separated_diagonals_vector_field.append(break_diagonal_vec([p1_x, p1_y, p2_x, p2_y])[1])
        else:
            separated_diagonals_vector_field.append([p1_x, p1_y, p2_x, p2_y])

    return separated_diagonals_vector_field


def is_vec_diagonal(vec):
    if np.abs(vec[0] - vec[2]) > epsilon:
        if np.abs(vec[1] - vec[3]) > epsilon:
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
        if np.abs(p1_x - p2_x) < epsilon:
            if p1_y < p2_y:
                up_vecs.append([p1_x, p1_y, p2_x, p2_y])
            else:
                down_vecs.append([p1_x, p1_y, p2_x, p2_y])
        if np.abs(p1_y - p2_y) < epsilon:
            if p1_x < p2_x:
                right_vecs.append([p1_x, p1_y, p2_x, p2_y])
            else:
                left_vecs.append([p1_x, p1_y, p2_x, p2_y])

    return up_vecs, down_vecs, right_vecs, left_vecs


def pair_vecs(up_vecs, down_vecs, right_vecs, left_vecs, boundaries, a):

    up = np.array(up_vecs)[:, [0, 1]].tolist()
    down = np.array(down_vecs)[:, [0, 1]].tolist()
    right = np.array(right_vecs)[:, [0, 1]].tolist()
    left = np.array(left_vecs)[:, [0, 1]].tolist()

    up_down_pairing = pairing_two_sides(up, down, boundaries, a)
    right_left_pairing = pairing_two_sides(right, left, boundaries, a)

    paired_up_down, unpaired_up_down = make_paired_Burger_field(up_vecs, down_vecs, up_down_pairing, 0)
    paired_right_left, unpaired_right_left = make_paired_Burger_field(right_vecs, left_vecs, right_left_pairing, len(paired_up_down))

    paired_Burgers_field = paired_up_down + paired_right_left

    return paired_Burgers_field, unpaired_up_down, unpaired_right_left


def second_optimization_pairing(paired_Burgers_field, unpaired_up_down, unpaired_right_left, boundaries, a):

    left = to_vecs(unpaired_up_down)
    right = to_vecs(unpaired_right_left)

    full_vecs = unpaired_up_down + unpaired_right_left

    pairing = pairing_two_sides(left, right, boundaries, a)

    for (u, v) in pairing:
        paired_Burgers_field[full_vecs[u][1]][1] = full_vecs[v][1]
        paired_Burgers_field[full_vecs[v][1]][1] = full_vecs[u][1]

    for [p1_x, p1_y, p2_x, p2_y], neighbor in paired_Burgers_field:
        if neighbor == -1:
            print([p1_x, p1_y, p2_x, p2_y])


def make_paired_Burger_field(first_side, second_side, pairing, offset):

    unpaired = []
    unpaired_no = 0
    full_vecs = first_side + second_side
    paired = [[[0]*4, -1]] * len(full_vecs)

    for (u, v) in pairing:
        paired[u] = [full_vecs[u], v + offset]
        paired[v] = [full_vecs[v], u + offset]

    for i in range(len(full_vecs)):
        if paired[i][1] == -1:
            unpaired_no += 1
            print(full_vecs[i])
            unpaired.append([full_vecs[i], i+offset])
            paired[i] = [full_vecs[i], -1]

    print("no of unpaired dislocations from this batch is", unpaired_no)

    return paired, unpaired


def pairing_two_sides(first_side, second_side, boundaries, a):

    """ using the following paper: https://dl.acm.org/doi/pdf/10.1145/6462.6502
    “Efficient Algorithms for Finding Maximum Matching in Graphs”, Zvi Galil, ACM Computing Surveys, 1986."""

    weighted_edges = []

    for i in range(len(first_side)):
        for j in range(len(second_side)):
            distance = utils.cyc_dist(first_side[i], second_side[j], boundaries)
            if distance < 5 * a:
                weighted_edges.append([i, len(first_side)+j, distance])

    G = nx.Graph()
    print("pairing up")
    G.add_weighted_edges_from(weighted_edges)
    pairing = nx.min_weight_matching(G)

    return pairing


def isolate_edges_that_cross_pairs(paired_Burgers_field, list_of_edges, boundaries, points, a):

    crossed_edges_indices = return_indices_of_edges_that_cross_Burgers_pair(list_of_edges, paired_Burgers_field, points,
                                                                            boundaries)

    for index in crossed_edges_indices:
        list_of_edges[index][2] = True


def return_indices_of_edges_that_cross_Burgers_pair(list_of_edges, paired_Burgers_field, points, boundaries):

    pairs_connecting_lines = []
    edges = []
    for [p1_x, p1_y, p2_x, p2_y], neighbor in paired_Burgers_field:
        p1 = (p1_x, p1_y)
        p2 = (paired_Burgers_field[neighbor][0][0], paired_Burgers_field[neighbor][0][1])
        if utils.vec_length_from_2_points(p1, p2) < boundaries[0] / 2:
            pair_line = LineString([p1, p2])
            pairs_connecting_lines.append(pair_line)

    for (u, v), color, in_circuit in list_of_edges:
        point1 = points[u]
        point2 = points[v]
        edge = LineString([point1, point2])
        edges.append(edge)

    pairs_connecting_lines_gpd = gpd.GeoDataFrame(geometry=pairs_connecting_lines)
    edges_gpd = gpd.GeoDataFrame(geometry=edges)

    crossings_gpd = gpd.sjoin(edges_gpd, pairs_connecting_lines_gpd, predicate='intersects')

    crossed_edges = list(set(crossings_gpd.index))

    return crossed_edges


def to_vecs(vec_with_index):
    vecs = []
    for [p1_x, p1_y, p2_x, p2_y], neighbor in vec_with_index:
        vecs.append([p1_x, p1_y, p2_x, p2_y])

    return vecs


"""

__________________________________________________________________________________________________________________
------------------------------------------------------------------------------------------------------------------

"cemetery of functions that fell out of use, to be deleted"

____________________________________________________________________________________________________________________
--------------------------------------------------------------------------------------------------------------------

"""


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


def connect_and_plot_pairs(first_side, second_side, pairing):

    two_sides = first_side + second_side

    print("plotting pairs")
    for (u, v) in pairing:
        if utils.vec_length_from_2_points([two_sides[u][0], two_sides[u][1]], [two_sides[v][0], two_sides[v][1]]) < 300:
            plt.plot([two_sides[u][0], two_sides[v][0]], [two_sides[u][1], two_sides[v][1]], color="purple")


def create_polygon(a, vec1, vec2):

    buffer = 1.2

    p1 = [vec1[0], vec1[1]]
    p2 = [vec2[0], vec2[1]]

    p11 = p1 + np.array([-buffer * a, buffer * a])
    p12 = p1 + np.array([-buffer * a, -buffer * a])

    p21 = p2 + np.array([buffer * a, buffer * a])
    p22 = p2 + np.array([buffer * a, -buffer * a])

    poly = Polygon([p11, p21, p22, p12])

    return poly


def is_point_in_polygon(polygon, points):
    no_of_points = len(points)
    in_polygon = np.full(no_of_points, False)

    for i in range(no_of_points):

        p = Point(points[i])
        in_polygon[i] = p.within(polygon)

    return in_polygon


def isolate_edges_that_cross_pairs_old(paired_Burgers_field, list_of_edges, boundaries, points):

    j = 0

    for [p1_x, p1_y, p2_x, p2_y], neighbor in paired_Burgers_field:
        j += 1
        if j % 50 == 0:
            print("isolating edges that cross Burgers vec pairs = ", int((j / len(paired_Burgers_field)) * 100), "%")
        if neighbor != -1:
            for i in range(len(list_of_edges)):
                p1 = [p1_x, p1_y]
                p2 = [paired_Burgers_field[neighbor][0][0],  paired_Burgers_field[neighbor][0][1]]
                p3 = points[list_of_edges[i][0][0]]
                p4 = points[list_of_edges[i][0][1]]

                if utils.do_2_lines_intersect(p1, p2, p3, p4):
                    if utils.vec_length_from_2_points(p1, p2) < boundaries[0]/2:
                        list_of_edges[i][2] = True


""" #################################################################################################################"""


def pair_vecs_old(up_vecs, down_vecs, right_vecs, left_vecs, boundaries, a):
    up_down_cost_matrix = calculate_cost_matrix(up_vecs, down_vecs, boundaries, a)
    right_left_cost_matrix = calculate_cost_matrix(right_vecs, left_vecs, boundaries, a)

    up_down_row_ind, up_down_col_ind = linear_sum_assignment(up_down_cost_matrix)
    right_left_row_ind, right_left_col_ind = linear_sum_assignment(right_left_cost_matrix)

    connect_and_plot_pairs_old(up_vecs, down_vecs, up_down_col_ind, up_down_row_ind, a)
    connect_and_plot_pairs_old(right_vecs, left_vecs, right_left_col_ind, right_left_row_ind, a)


def calculate_cost_matrix(first_side, second_side, boundaries, a):
    cost_matrix = np.zeros((len(first_side), len(second_side)))

    for i in range(len(first_side)):
        for j in range(len(second_side)):
            distance = utils.cyc_dist(first_side[i], second_side[j], boundaries)
            cost_matrix[i][j] = np.exp(distance)

    return cost_matrix


def connect_and_plot_pairs_old(first_side, second_side, col_ind, row_ind, a):
    j = 0
    for i in range(len(row_ind)):
        if utils.vec_length_from_2_points([first_side[row_ind[i]][0], first_side[row_ind[i]][1]],
                                          [second_side[col_ind[i]][0], second_side[col_ind[i]][1]]) < 300:

            plt.plot([first_side[row_ind[i]][0], second_side[col_ind[i]][0]],
                     [first_side[row_ind[i]][1], second_side[col_ind[i]][1]], color="purple")

            if utils.vec_length_from_2_points([first_side[row_ind[i]][0], first_side[row_ind[i]][1]],
                                              [second_side[col_ind[i]][0], second_side[col_ind[i]][1]]) > 5 * a:
                j += 1
                print(j)


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

    # create_list_of_polygons(a, Burger_field)
    pass


def add_up_to_zero(vec1, vec2):
    sum = utils.two_points_sum(utils.two_points_2_vector(vec1[0][0:2], vec1[0][2:4]),
                               utils.two_points_2_vector(vec2[0][0:2], vec2[0][2:4]))
    for i in sum:
        if np.abs(i) > epsilon:
            return False

    return True


def less_first(vec1, vec2):
    p1 = np.array(midpoint(vec1[0:2], vec1[2:4]))
    p2 = np.array(midpoint(vec2[0:2], vec2[2:4]))

    if p1[0] > p2[0]:
        return p2, p1
    return p1, p2


def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
