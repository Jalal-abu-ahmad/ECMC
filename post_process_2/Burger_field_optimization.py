import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
from shapely.geometry import Point, Polygon

from post_process_2 import utils

epsilon = 0.00001


def Burger_vec_pairing(points, list_of_edges, Burger_field, a):
    Burger_vecs_centers = []
    for [p1_x, p1_y, p2_x, p2_y], neighbor in Burger_field:
        Burger_vecs_centers.append(midpoint([p1_x, p1_y], [p2_x, p2_y]))
    iterated_pairing(Burger_field, Burger_vecs_centers, no_of_iterations=6)
    isolate_unsatisfied_paths(Burger_field, Burger_vecs_centers, list_of_edges, points, a)


def midpoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]


def iterated_pairing(Burger_field, Burger_vecs_centers, no_of_iterations):

    unpaired_vecs = []
    for i in range(len(Burger_vecs_centers)):
        unpaired_vecs.append([Burger_vecs_centers[i], int(i)])
    for iteration in range(no_of_iterations):
        pairs_only = [unpaired_vecs[i][0] for i in range(len(unpaired_vecs))]
        NNgraph = kneighbors_graph(pairs_only, n_neighbors=1)
        check_and_arrange_pairing(NNgraph, Burger_field, unpaired_vecs)
        unpaired_vecs = clean_up(Burger_field, unpaired_vecs)


def check_and_arrange_pairing(NNgraph, Burger_field, unpaired_vecs):
    # check if:
    # 1) the closest neighbors for a pair (if A is NN of b then B is NN of A)
    # 2) check if the pairs cancel out
    # -> pair up the vectors that satisfy 1) & 2)
    nearest_neighbor = NNgraph.indices

    for i in range(len(nearest_neighbor)):
        if nearest_neighbor[nearest_neighbor[i]] == i:
            if add_up_to_zero(Burger_field[i], Burger_field[nearest_neighbor[i]]):
                Burger_field[unpaired_vecs[i][1]][1] = unpaired_vecs[nearest_neighbor[i]][1]
                Burger_field[unpaired_vecs[nearest_neighbor[i]][1]][1] = unpaired_vecs[i][1]


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

    create_list_of_polygons(points, a, Burger_field)


def add_up_to_zero(vec1, vec2):
    sum = utils.two_points_sum(utils.two_points_2_vector(vec1[0][0:2], vec1[0][2:4]),
                               utils.two_points_2_vector(vec2[0][0:2], vec2[0][2:4]))
    for i in sum:
        if np.abs(i) > epsilon:
            return False

    return True


def create_list_of_polygons(list_of_points, a, Burger_field):

    list_of_polygons = []

    for [p1_x, p1_y, p2_x, p2_y], neighbor in Burger_field:
        if neighbor != -1:
            p1, p2 = less_first(Burger_field[neighbor][0], [p1_x, p1_y, p2_x, p2_y])
            '''plot a line between p1, and p2'''
            p11 = p1 + np.array([-1.5*a, 1.5*a])
            p12 = p1 + np.array([-1.5*a, -1.5*a])

            p21 = p2 + np.array([1.5*a, 1.5*a])
            p22 = p2 + np.array([1.5*a, -1.5*a])

            list_of_polygons.append(Polygon([p11, p12, p21, p22]))

    return list_of_polygons


def less_first(vec1, vec2):
    p1 = np.array(midpoint(vec1[0:2], vec1[2:4]))
    p2 = np.array(midpoint(vec2[0:2], vec2[2:4]))

    if p1[0] > p2[0]:
        return p2, p1
    return p1, p2
