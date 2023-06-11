from scipy.spatial import Delaunay
import matplotlib as plt
import utils
import numpy as np


def Burger_field_calculation(points,l_x, l_y, N, global_theta,a):

    Burger_field=[]
    tri = Delaunay(points)
    perfect_lattice_diagonal_vecs, perfect_lattice_non_diagonal_vecs = utils.perfect_lattice_vectors(a,1)
    perfect_lattice_vecs = np.append(perfect_lattice_non_diagonal_vecs, perfect_lattice_non_diagonal_vecs)
    aligned_perfect_lattice_vecs = utils.rotate_points_by_angle(perfect_lattice_vecs, global_theta)
    for i, triangle in enumerate(tri.simplices):
        ab_ref,bc_ref,ca_ref=compare_triangle_edges_to_referace_lattice(triangle,aligned_perfect_lattice_vecs)
        Burger_circut = edge2vector(ab_ref) + edge2vector(bc_ref) + edge2vector(ca_ref)
        if Burger_circut != [0,0]:
            Burger_field.append(Burger_vector_calc(points[triangle[0]],Burger_circut))


    return Burger_field

    """
    deal with cyclic boundry conditions
    """

def edge2vector(edge):
  return np.array([edge[1]-edge[0]])

def Burger_vector_calc(origin_point,Burger_circut):

    Burger_vector=([origin_point[0], origin_point[1], Burger_circut[0]+origin_point[0],Burger_circut[1]+origin_point[1]])

    return Burger_vector
def compare_triangle_edges_to_referace_lattice(points,triangle, reference_lattice_vecs):
    ab=np.array([points[triangle[0]],points[triangle[1]]])
    bc=np.array([points[triangle[1]],points[triangle[2]]])
    ca=np.array([points[triangle[2]],points[triangle[0]]])
    ab_ref = closest_reference_vector(ab, reference_lattice_vecs)
    bc_ref = closest_reference_vector(bc, reference_lattice_vecs)
    ca_ref = closest_reference_vector(ca, reference_lattice_vecs)

    return ab_ref,bc_ref,ca_ref

def closest_reference_vector(vec_ab,reference_lattice):
    i = np.argmin([np.linalg.norm(vec_ab - L) for L in reference_lattice])
    return np.array([reference_lattice[i]])

