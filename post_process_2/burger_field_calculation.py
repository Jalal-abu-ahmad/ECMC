import numpy as np
from scipy.spatial import Delaunay

import utils

epsilon = 0.00001


def Burger_field_calculation(points, l_x, l_y, N, global_theta, a, order):
    Burger_field = [[0, 0, 0, 0]]
    list_of_edges = []
    tri = Delaunay(points)
    triangle_mid_points = tri.points[tri.simplices].mean(axis=1)
    no_of_triangles = len(tri.simplices)
    perfect_lattice_diagonal_vecs, perfect_lattice_non_diagonal_vecs = utils.perfect_lattice_vectors(a, order)
    aligned_perfect_lattice_diag_vecs = utils.rotate_points_by_angle(perfect_lattice_diagonal_vecs, global_theta)
    aligned_perfect_lattice_non_diag_vecs = utils.rotate_points_by_angle(perfect_lattice_non_diagonal_vecs,
                                                                         global_theta)
    for i, triangle in enumerate(tri.simplices):
        if (i % 1000) == 0:
            print("Burger field progress = ", int((i / no_of_triangles) * 100), "%")
        ab_ref, bc_ref, ca_ref = sort_triangle_edges_compared_to_reference_lattice(points, triangle,
                                                                                   aligned_perfect_lattice_non_diag_vecs,
                                                                                   aligned_perfect_lattice_diag_vecs,
                                                                                   list_of_edges)

        Burger_circuit = ab_ref + bc_ref + ca_ref
        if is_not_zero(Burger_circuit):
            Burger_field = np.row_stack((Burger_field, Burger_vector_calc(triangle_mid_points[i], Burger_circuit)))
    Burger_field = np.delete(Burger_field, 0, 0)
    return np.array(Burger_field), list_of_edges

    """
    deal with cyclic boundary conditions
    """


def is_not_zero(Burger_circuit):
    for coor in Burger_circuit:
        for i in coor:
            if np.abs(i) > epsilon:
                return True

    return False


def edge2vector(edge, points):
    return np.array([points[edge[1]] - points[edge[0]]])


def Burger_vector_calc(triangle_mid_point, Burger_circuit):
    Burger_vector = (triangle_mid_point[0], triangle_mid_point[1], Burger_circuit[0][0] + triangle_mid_point[0],
                     Burger_circuit[0][1] + triangle_mid_point[1])

    return Burger_vector


def sort_triangle_edges_compared_to_reference_lattice(points, triangle,
                                                      aligned_perfect_lattice_non_diag_vecs,
                                                      aligned_perfect_lattice_diag_vecs, list_of_edges):

    reference_lattice_vecs = np.row_stack((aligned_perfect_lattice_non_diag_vecs, aligned_perfect_lattice_diag_vecs))

    ab = np.array([triangle[0], triangle[1]])
    bc = np.array([triangle[1], triangle[2]])
    ca = np.array([triangle[2], triangle[0]])
    ab_ref = closest_reference_vector(edge2vector(ab, points), reference_lattice_vecs)
    bc_ref = closest_reference_vector(edge2vector(bc, points), reference_lattice_vecs)
    ca_ref = closest_reference_vector(edge2vector(ca, points), reference_lattice_vecs)

    for e in [ab, bc, ca]:
        if utils.is_diagonal(e, aligned_perfect_lattice_diag_vecs, aligned_perfect_lattice_non_diag_vecs, points):
            list_of_edges.append((e, "red"))
        else:
            if utils.is_horizontal(e, points):
                list_of_edges.append((e, "y"))
            else:
                list_of_edges.append((e, "blue"))

    return ab_ref, bc_ref, ca_ref


def closest_reference_vector(vec_ab, reference_lattice):
    i = np.argmin([np.linalg.norm(vec_ab - L) for L in reference_lattice])
    return np.array([reference_lattice[i]])
