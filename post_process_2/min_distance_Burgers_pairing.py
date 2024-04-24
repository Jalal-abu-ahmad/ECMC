import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix

zero_mask = 10000


def make_pairs_distances_list(Burgers_field):

    paired_distances = []
    distance_mat = distance_matrix_calc(Burgers_field)
    dim = len(distance_mat)
    while dim > 1:
        min_element, distance_mat = find_and_delete_min_row_col(distance_mat)
        paired_distances.append(min_element)
        dim = len(distance_mat)

    plt.hist(paired_distances)
    plt.show()
    return paired_distances


def distance_matrix_calc(Burgers_field):
    N = len(Burgers_field)
    distance_mat = np.matrix(distance_matrix(Burgers_field, Burgers_field))
    mask = np.ones((N, N)).astype(bool)  # same shape as the array
    mask = np.logical_and(mask, distance_mat == 0)
    distance_mat[mask] = zero_mask

    return distance_mat


def find_and_delete_min_row_col(distance_mat):

    min_ind = np.unravel_index(np.argmin(distance_mat, axis=None), distance_mat.shape)
    min_element = distance_mat[min_ind]
    distance_mat = np.delete(distance_mat, [min_ind[0], min_ind[1]], 0)
    distance_mat = np.delete(distance_mat, [min_ind[0], min_ind[1]], 1)

    return min_element, distance_mat


if __name__ == "__main__":
    field = [[0,1], [4,5], [8,9], [1,7], [5,3]]

    make_pairs_distances_list(field)