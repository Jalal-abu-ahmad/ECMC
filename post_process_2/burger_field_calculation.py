import numpy as np
from scipy.spatial import Delaunay

import utils

epsilon = 0.00001


def Burger_field_calculation(points, a, order):
    Burger_field = [[0, 0, 0, 0]]
    list_of_edges = []
    tri = Delaunay(points)
    triangle_mid_points = tri.points[tri.simplices].mean(axis=1)
    no_of_triangles = len(tri.simplices)
    perfect_lattice_diagonal_vecs, perfect_lattice_non_diagonal_vecs = utils.perfect_lattice_vectors(a, order)
    for i, triangle in enumerate(tri.simplices):
        if (i % 1000) == 0:
            print("Burger field progress = ", int((i / no_of_triangles) * 100), "%")
        ab_ref, bc_ref, ca_ref = sort_triangle_edges_compared_to_reference_lattice(points, triangle,
                                                                                     perfect_lattice_non_diagonal_vecs,
                                                                                     perfect_lattice_diagonal_vecs,
                                                                                     list_of_edges)

        Burger_circuit = ab_ref + bc_ref + ca_ref
        if is_not_zero(Burger_circuit):
            Burger_field = np.row_stack((Burger_field, Burger_vector_calc(triangle_mid_points[i], Burger_circuit)))
            Burger_points_and_edges(points, list_of_edges, triangle)
    Burger_field = np.delete(Burger_field, 0, 0)
    return np.array(Burger_field), list_of_edges


def Burger_points_and_edges(points, list_of_edges, triangle):
    for i in [-1, -2, -3]:
        list_of_edges[i][2] = True


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

    ab = [triangle[0], triangle[1]]
    bc = [triangle[1], triangle[2]]
    ca = [triangle[2], triangle[0]]
    ab_ref = closest_reference_vector(edge2vector(ab, points), reference_lattice_vecs)
    bc_ref = closest_reference_vector(edge2vector(bc, points), reference_lattice_vecs)
    ca_ref = closest_reference_vector(edge2vector(ca, points), reference_lattice_vecs)

    for e in [ab, bc, ca]:
        if not found_edge(e, list_of_edges):
            if utils.is_diagonal(e, aligned_perfect_lattice_diag_vecs, aligned_perfect_lattice_non_diag_vecs, points):
                # false means that the edge is not in a non-zero Burger-Circuit, will be overriden later if edge is in a
                # non zero burger circuit
                list_of_edges.append([e, "red", False])
            else:
                if utils.is_horizontal(e, points):
                    list_of_edges.append([e, "y", False])
                else:
                    list_of_edges.append([e, "blue", False])
    return ab_ref, bc_ref, ca_ref


def closest_reference_vector(vec_ab, reference_lattice):
    i = np.argmin([np.linalg.norm(vec_ab - L) for L in reference_lattice])
    return np.array([reference_lattice[i]])


def found_edge(edge, list_0f_edges) -> bool:
    for (p1, p2), color, in_circuit in list_0f_edges:
        if matching_edge([p1, p2], edge):
            return True
    return False


def matching_edge(edge1, edge2):
    p1, p2 = less_first(edge1[0], edge1[1])
    d1, d2 = less_first(edge2[0], edge2[1])

    if p1 == d1 and p2 == d2:
        return True
    return False


def less_first(a, b):
    if a > b:
        return b, a
    return a, b
