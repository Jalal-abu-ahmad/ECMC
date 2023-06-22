from scipy.spatial import Delaunay
import matplotlib as plt
import utils
import numpy as np

epsilon = 0.00001


def Burger_field_calculation(points, l_x, l_y, N, global_theta, a):
    Burger_field = [[0, 0, 0, 0]]
    list_of_edges = []
    tri = Delaunay(points)
    triangle_mid_points = tri.points[tri.simplices].mean(axis=1)
    no_of_triangles = len(tri.simplices)
    perfect_lattice_diagonal_vecs, perfect_lattice_non_diagonal_vecs = utils.perfect_lattice_vectors(a, 1)
    # perfect_lattice_vecs = np.row_stack((perfect_lattice_non_diagonal_vecs, perfect_lattice_diagonal_vecs))
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

        Burger_circut = ab_ref + bc_ref + ca_ref
        if is_not_zero(Burger_circut):
            Burger_field = np.row_stack((Burger_field, Burger_vector_calc(triangle_mid_points[i], Burger_circut)))
    Burger_field = np.delete(Burger_field, 0, 0)
    return np.array(Burger_field), list_of_edges

    """
    deal with cyclic boundry conditions
    """


def is_not_zero(Burger_circut):
    for coor in Burger_circut:
        for i in coor:
            if np.abs(i) > epsilon:
                return True

    return False


def edge2vector(edge):
    return np.array([edge[1] - edge[0]])


def Burger_vector_calc(triangle_mid_point, Burger_circut):
    Burger_vector = (triangle_mid_point[0], triangle_mid_point[1], Burger_circut[0][0] + triangle_mid_point[0],
                     Burger_circut[0][1] + triangle_mid_point[1])

    return Burger_vector


def sort_triangle_edges_compared_to_reference_lattice(points, triangle,
                                                      aligned_perfect_lattice_non_diag_vecs,
                                                      aligned_perfect_lattice_diag_vecs, list_of_edges):

    reference_lattice_vecs = np.row_stack((aligned_perfect_lattice_non_diag_vecs, aligned_perfect_lattice_diag_vecs))

    ab = np.array([points[triangle[0]], points[triangle[1]]])
    bc = np.array([points[triangle[1]], points[triangle[2]]])
    ca = np.array([points[triangle[2]], points[triangle[0]]])
    ab_ref = closest_reference_vector(edge2vector(ab), reference_lattice_vecs)
    bc_ref = closest_reference_vector(edge2vector(bc), reference_lattice_vecs)
    ca_ref = closest_reference_vector(edge2vector(ca), reference_lattice_vecs)

    for e in [ab, bc, ca]:
        if utils.is_diagonal(e, aligned_perfect_lattice_diag_vecs, aligned_perfect_lattice_non_diag_vecs,points):
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
