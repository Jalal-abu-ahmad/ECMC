import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph

from post_process_2 import utils

epsilon = 0.00001


def Burger_vec_pairing(points, list_of_edges, Burger_field, a):
    Burger_vecs_centers = []
    for p1_x, p1_y, p2_x, p2_y, neighbor in Burger_field:
        Burger_vecs_centers.append(midpoint([p1_x, p1_y], [p2_x, p2_y]))

    iterated_pairing(Burger_field, Burger_vecs_centers, no_of_iterations=4)
    isolate_unsatisfied_paths(Burger_field, Burger_vecs_centers, list_of_edges, points, a)


def midpoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]


def iterated_pairing(Burger_field, Burger_vecs_centers, no_of_iterations):

    unpaired_vecs = np.array(Burger_vecs_centers)
    unpaired_vecs = np.column_stack([unpaired_vecs, np.array(range(0, len(unpaired_vecs)))])
    for iteration in range(no_of_iterations):
        NNgraph = kneighbors_graph(unpaired_vecs[:, [0, 1]], n_neighbors=1)
        check_and_arrange_pairing(NNgraph, Burger_field, unpaired_vecs)
        unpaired_vecs = clean_up(Burger_field, unpaired_vecs)


def check_and_arrange_pairing(NNgraph, Burger_field, unpaired_vecs):
    # check if:
    # 1) the closest neighbors for a pair (if A is NN of b then B is NN of A)
    # 2) check if the pairs cancel out
    # -> pair up the vectors that satisfy 1) & 2)
    nearest_neighbor = get_nearest_neighbor(len(unpaired_vecs), NNgraph)

    for i in range(len(nearest_neighbor)):
        if nearest_neighbor[nearest_neighbor[i]] == i:
            if add_up_to_zero(Burger_field[i], Burger_field[nearest_neighbor[i]]):
                Burger_field[unpaired_vecs[i][2]][4] = nearest_neighbor[i]
                Burger_field[unpaired_vecs[nearest_neighbor[i]][2]][4] = i


def clean_up(Burgers_field, last_iteration_of_pairing):
    # remove approved pairs and keep unpaired vecs to start next iteration

    unpaired_vecs = []
    for p_x, p_y, original_index in last_iteration_of_pairing:
        if Burgers_field[original_index][1] == -1:
            unpaired_vecs = np.append(unpaired_vecs, [[p_x, p_y, original_index]])

    return unpaired_vecs


def isolate_unsatisfied_paths(Burger_field, Burger_vecs_centers, list_of_edges, points, a):
    # isolate all bonds and points between pairs as they are in a non-neutral zone
    # use : "https://www.matecdev.com/posts/point-in-polygon.html"
    pass


def add_up_to_zero(vec1, vec2):
    sum = utils.two_points_2_vector(utils.two_points_2_vector(vec1[0], vec2[1]), utils.two_points_2_vector(vec1[0],
                                                                                                           vec2[1]))
    for i in sum:
        if np.abs(i) > epsilon:
            return False

    return True


def create_list_of_polygons(list_of_points):
    pass


def get_nearest_neighbor(N, NNgraph):
    return [NNgraph.getrow(i).indices for i in range(N)]
