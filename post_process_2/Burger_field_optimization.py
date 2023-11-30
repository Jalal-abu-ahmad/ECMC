import geopandas as gpd
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon, LineString
from sklearn.neighbors import kneighbors_graph
import utils

epsilon = 0.00001


def Burger_vec_optimization(points, list_of_edges, Burger_field, a, boundaries, theta):

    no_duplicates_Burger_field = clean_Burger_field(Burger_field, boundaries, theta)
    Burger_field_diagonals_separated = break_diagonal_vecs_2_components(no_duplicates_Burger_field)

    paired_Burgers_field, unpaired_vecs = pair_vec_of_all_kinds(Burger_field_diagonals_separated, boundaries, a, theta)
    pairs_connecting_lines = isolate_edges_that_cross_pairs(paired_Burgers_field, list_of_edges, boundaries, points, a, theta)

    print("no of dislocations=", len(Burger_field_diagonals_separated))

    Burgers_params = map(str, [len(Burger_field_diagonals_separated), len(unpaired_vecs)])
    return paired_Burgers_field, pairs_connecting_lines, Burgers_params


def clean_Burger_field(Burger_field, boundaries, theta):

    first_rotation = utils.rotate_Burger_vecs(Burger_field, -theta)
    remove_duplicates = utils.remove_points_outside_boundaries(first_rotation, boundaries)
    back_to_normal = utils.rotate_Burger_vecs(remove_duplicates, theta)

    return back_to_normal


def break_diagonal_vecs_2_components(vector_field):

    separated_diagonals_vector_field = []
    count =0

    for [p1_x, p1_y, p2_x, p2_y] in vector_field:
        if is_vec_diagonal([p1_x, p1_y, p2_x, p2_y]):
            # separated_diagonals_vector_field.append(break_diagonal_vec([p1_x, p1_y, p2_x, p2_y])[0])
            # separated_diagonals_vector_field.append(break_diagonal_vec([p1_x, p1_y, p2_x, p2_y])[1])
            count = count + 1
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


def pair_vec_of_all_kinds(all_vecs_full, boundaries, a, theta):

    all_vecs = np.array(all_vecs_full)[:, [0, 1]].tolist()
    pairing = pairing_two_sides_all_options(all_vecs, all_vecs_full, boundaries, a, theta, 5)

    paired, unpaired = make_paired_Burger_field_all_options(all_vecs_full, pairing)

    second_pairing_points = to_vecs(unpaired)

    second_pairing = pairing_two_sides_all_options(second_pairing_points, all_vecs_full, boundaries, a, theta, 300)

    for (u, v) in second_pairing:
        paired[unpaired[u][1]][1] = unpaired[v][1]
        paired[unpaired[v][1]][1] = unpaired[u][1]

    return paired, unpaired


def make_paired_Burger_field_all_options(all_vecs_full, pairing):

    unpaired = []
    unpaired_no = 0
    paired = [[[0]*4, -1]] * len(all_vecs_full)

    for (u, v) in pairing:
        paired[u] = [all_vecs_full[u], v]
        paired[v] = [all_vecs_full[v], u]

    for i in range(len(all_vecs_full)):
        if paired[i][1] == -1:
            unpaired_no += 1
            print(all_vecs_full[i])
            unpaired.append([all_vecs_full[i], i])
            paired[i] = [all_vecs_full[i], -1]

    print("no of unpaired dislocations from this batch is", unpaired_no)

    return paired, unpaired


def pairing_two_sides_all_options(all_vecs, all_vecs_full, boundaries, a, theta, coeff):

    """ using the following paper: https://dl.acm.org/doi/pdf/10.1145/6462.6502
    “Efficient Algorithms for Finding Maximum Matching in Graphs”, Zvi Galil, ACM Computing Surveys, 1986."""

    weighted_edges = []

    first_side = utils.rotate_points_by_angle(all_vecs, -theta)
    second_side = utils.rotate_points_by_angle(all_vecs, -theta)
    full = utils.rotate_Burger_vecs(all_vecs_full, -theta)

    for i in range(len(first_side)):
        for j in range(len(second_side)):
            distance = utils.cyc_dist(first_side[i], second_side[j], boundaries)
            if coeff * a > distance > 0 and not_same_point(first_side[i], second_side[j]):
                weighted_edges.append([i, j, distance])

    G = nx.Graph()
    print("pairing up")
    G.add_weighted_edges_from(weighted_edges)
    pairing = nx.min_weight_matching(G)

    return pairing


def not_same_point(p1, p2):

    if p1[0] == p2[0] and p1[1] == p2[1]:
        return False
    return True


def isolate_edges_that_cross_pairs(paired_Burgers_field, list_of_edges, boundaries, points, a, theta):

    crossed_edges_indices, pairs_connecting_lines = return_indices_of_edges_that_cross_Burgers_pair(list_of_edges, paired_Burgers_field, points,
                                                                            boundaries, theta)

    for index in crossed_edges_indices:
        list_of_edges[index][2] = True

    return pairs_connecting_lines


def return_indices_of_edges_that_cross_Burgers_pair(list_of_edges, paired_Burgers_field, points, boundaries, theta):

    pairs_connecting_lines = []
    edges = []
    for [p1_x, p1_y, p2_x, p2_y], neighbor in paired_Burgers_field:
        if neighbor != -1:
            p1 = (p1_x, p1_y)
            p2 = (paired_Burgers_field[neighbor][0][0], paired_Burgers_field[neighbor][0][1])
            if not utils.paired_through_boundary(p1, p2, boundaries, theta):
                pair_line = LineString([p1, p2])
                pairs_connecting_lines.append(pair_line)
            else:
                pair_line_1, pair_line_2 = pair_points_through_boundary(p1, p2, boundaries, theta)
                pairs_connecting_lines.append(pair_line_1)
                pairs_connecting_lines.append(pair_line_2)

    for (u, v), color, in_circuit in list_of_edges:
        point1 = points[u]
        point2 = points[v]
        edge = LineString([point1, point2])
        edges.append(edge)

    pairs_connecting_lines_gpd = gpd.GeoDataFrame(geometry=pairs_connecting_lines)
    edges_gpd = gpd.GeoDataFrame(geometry=edges)

    crossings_gpd = gpd.sjoin(edges_gpd, pairs_connecting_lines_gpd, predicate='intersects')

    crossed_edges = list(set(crossings_gpd.index))

    return crossed_edges, pairs_connecting_lines


def pair_points_through_boundary(p1, p2, boundaries, theta):
    p1, p2 = utils.rotate_points_by_angle([p1, p2], -theta)
    which_boundary = [0, 0]

    dx = np.array(p1) - p2  # direct vector
    for i in range(2):
        L = boundaries[i]
        if (dx[i] + L) ** 2 < dx[i] ** 2:
            which_boundary[i] = L
        if (dx[i] - L) ** 2 < dx[i] ** 2:
            which_boundary[i] = -L

    p1_b = p1 + which_boundary
    p2_b = p2 - which_boundary
    boundary_1 = intersect_with_boundaries(p1, p2_b, boundaries)
    boundary_2 = intersect_with_boundaries(p1_b, p2, boundaries)

    p1, p2, b1, b2 = utils.rotate_points_by_angle([p1, p2, boundary_1, boundary_2], theta)
    line_1 = LineString([p1, b1])
    line_2 = LineString([p2, b2])

    return line_1, line_2


def intersect_with_boundaries(p1, p2, boundaries):

    boundary_pairs = [[[0, 0], [0, boundaries[1]]],
                      [[0, 0], [boundaries[0], 0]],
                      [[0, boundaries[1]], [boundaries[0], boundaries[1]]],
                      [[boundaries[0], 0], [boundaries[0], boundaries[1]]]]

    for pair in boundary_pairs:
        if utils.where_2_lines_intersect(p1, p2, pair[0], pair[1]):
            return utils.where_2_lines_intersect(p1, p2, pair[0], pair[1])


def to_vecs(vec_with_index):
    vecs = []
    for [p1_x, p1_y, p2_x, p2_y], original_index in vec_with_index:
        vecs.append([p1_x, p1_y])

    return vecs


"""

__________________________________________________________________________________________________________________
------------------------------------------------------------------------------------------------------------------

"cemetery of functions that fell out of use, to be deleted"

____________________________________________________________________________________________________________________
--------------------------------------------------------------------------------------------------------------------

"""


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


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


def pair_vecs(up_vecs, down_vecs, right_vecs, left_vecs, boundaries, a, theta):

    up = np.array(up_vecs)[:, [0, 1]].tolist()
    down = np.array(down_vecs)[:, [0, 1]].tolist()
    right = np.array(right_vecs)[:, [0, 1]].tolist()
    left = np.array(left_vecs)[:, [0, 1]].tolist()

    up_down_pairing = pairing_two_sides(up, down, boundaries, a, theta, 5)
    right_left_pairing = pairing_two_sides(right, left, boundaries, a, theta, 5)

    paired_up_down, unpaired_up_down = make_paired_Burger_field(up_vecs, down_vecs, up_down_pairing, 0)
    paired_right_left, unpaired_right_left = make_paired_Burger_field(right_vecs, left_vecs, right_left_pairing, len(paired_up_down))

    paired_Burgers_field = paired_up_down + paired_right_left

    return paired_Burgers_field, unpaired_up_down, unpaired_right_left


def second_optimization_pairing(paired_Burgers_field, unpaired_up_down, unpaired_right_left, boundaries, a, theta):

    up_down = to_vecs(unpaired_up_down)
    right_left = to_vecs(unpaired_right_left)
    unpaired_after_second_optimization = []

    vec_starting_points = up_down + right_left

    full_vecs = unpaired_up_down + unpaired_right_left

    second_pairing = pairing_two_sides_second_optimization(vec_starting_points, full_vecs, boundaries, a, theta, 60)

    for (u, v) in second_pairing:
        paired_Burgers_field[full_vecs[u][1]][1] = full_vecs[v][1]
        paired_Burgers_field[full_vecs[v][1]][1] = full_vecs[u][1]

    for idx, [[p1_x, p1_y, p2_x, p2_y], neighbor] in enumerate(paired_Burgers_field):
        if neighbor == -1:
            print([p1_x, p1_y, p2_x, p2_y])
            unpaired_after_second_optimization.append([[p1_x, p1_y, p2_x, p2_y], idx])

    third_optimization_starting_points = to_vecs(unpaired_after_second_optimization)
    third_pairing = pairing_two_sides_second_optimization(third_optimization_starting_points, unpaired_after_second_optimization, boundaries, a, theta, 200)

    for (u, v) in third_pairing:
        paired_Burgers_field[unpaired_after_second_optimization[u][1]][1] = unpaired_after_second_optimization[v][1]
        paired_Burgers_field[unpaired_after_second_optimization[v][1]][1] = unpaired_after_second_optimization[u][1]


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


def not_same_direction(vec1, vec2):

    first_vec_x = vec1[2] - vec1[0]
    first_vec_y = vec1[3] - vec1[1]
    second_vec_x = vec2[2] - vec2[0]
    second_vec_y = vec2[3] - vec2[1]

    if utils.dot_product([first_vec_x, first_vec_y], [second_vec_x, second_vec_y]) > 0:
        return False

    return True


def pairing_two_sides(first_side, second_side, boundaries, a, theta, coeff):

    """ using the following paper: https://dl.acm.org/doi/pdf/10.1145/6462.6502
    “Efficient Algorithms for Finding Maximum Matching in Graphs”, Zvi Galil, ACM Computing Surveys, 1986."""

    weighted_edges = []

    first_side = utils.rotate_points_by_angle(first_side, -theta)
    second_side = utils.rotate_points_by_angle(second_side, -theta)

    for i in range(len(first_side)):
        for j in range(len(second_side)):
            distance = utils.cyc_dist(first_side[i], second_side[j], boundaries)
            if coeff * a > distance > 0:
                weighted_edges.append([i, len(first_side)+j, distance])

    G = nx.Graph()
    print("pairing up")
    G.add_weighted_edges_from(weighted_edges)
    pairing = nx.min_weight_matching(G)

    return pairing


def pairing_two_sides_second_optimization(points, full_vec, boundaries, a, theta, coeff):

    """ using the following paper: https://dl.acm.org/doi/pdf/10.1145/6462.6502
    “Efficient Algorithms for Finding Maximum Matching in Graphs”, Zvi Galil, ACM Computing Surveys, 1986."""

    weighted_edges = []

    first_side = utils.rotate_points_by_angle(points, -theta)
    second_side = utils.rotate_points_by_angle(points, -theta)
    full = utils.rotate_Burger_vecs([full_vec[i][0] for i in range(len(full_vec))], -theta)

    for i in range(len(first_side)):
        for j in range(len(second_side)):
            distance = utils.cyc_dist(first_side[i], second_side[j], boundaries)
            if coeff * a > distance > 0 and not_same_point(first_side[i], second_side[j]):
                weighted_edges.append([i, j, distance])

    G = nx.Graph()
    print("pairing up")
    G.add_weighted_edges_from(weighted_edges)
    pairing = nx.min_weight_matching(G)

    return pairing

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
