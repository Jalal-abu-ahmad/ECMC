from post_process_2 import utils
import matplotlib.pyplot as plt
from post_process_2 import Graph
import numpy as np


def connectivity_Bipartiteness_AFism(list_of_edges, points_with_z, boundaries, theta, l_z):
    G = Graph.Graph()

    vertices, edges = keep_within_boundaries_and_non_isolated(G, list_of_edges, points_with_z, boundaries, theta)

    visited = check_connectivity(G, vertices, edges)
    vertices_sign = check_bipartiteness(G, vertices, edges, visited)
    calculate_AF_order_parameter(G, vertices, edges, vertices_sign, visited, l_z)


def keep_within_boundaries_and_non_isolated(G, list_of_edges, points, boundaries, theta):
    edges = []
    N = len(points)
    vertices = points
    # vertices = np.delete(points, 2, axis=1)
    # vertices = vertices.tolist()
    # first_rotation = utils.rotate_points_by_angle(points_without_z, -theta)

    for (u, v), color, in_circuit in list_of_edges:
        if not (in_circuit or color == 'red'):
            if u < N and v < N:
                G.addEdge(u, v)
                edges.append([u, v])

        # if not (in_circuit or color == 'red'):
        #     if not utils.out_of_boundaries(first_rotation[u], boundaries[0]):
        #         vertices.append(points[u])
        #         if not utils.out_of_boundaries(first_rotation[v], boundaries[1]):
        #             vertices.append(points[v])
        #             edges.append([len(vertices)-2, len(vertices)-1])
        #             G.addEdge(len(vertices)-2, len(vertices)-1)
        #     else:
        #         if not utils.out_of_boundaries(first_rotation[v], boundaries[1]):
        #             vertices.append(points[v])

    return vertices, edges


def check_connectivity(G, vertices, edges):
    i = 0
    visited = [G.BFS(0)]
    node = not_all_visited(visited)

    while node != -1:
        i += 1
        visited.append(G.BFS(node))
        node = not_all_visited(visited)

    print("Graph has", len(visited), "connected componentes")

    return visited


def not_all_visited(visited):
    full_visited = visited[0]

    for i in range(len(visited)):
        full_visited = [full_visited[j] or visited[i][j] for j in range(len(visited[0]))]

    for node in range(len(full_visited)):
        if not full_visited[node]:
            return node
    return -1


def check_bipartiteness(G, vertices, edges, visited):

    non_compatible_0, sign_0, visited_0 = G.check_if_Bipartite_BFS(0)
    visited = [visited_0]
    node = not_all_visited(visited)
    sign = [sign_0]
    print("non =", non_compatible_0)

    while node != -1:
        non_compatible_i, sign_i, visited_i = G.check_if_Bipartite_BFS(node)
        visited.append(visited_i)
        sign.append(sign_i)
        print("non =", non_compatible_i)
        node = not_all_visited(visited)

    return sign


def calculate_AF_order_parameter(G, vertices, edges, vertices_sign, visited, l_z):

    order_parameter, visited_0 = G.calculate_AF_order_parameter(0, 0, vertices, l_z)
    visited = [visited_0]
    node = not_all_visited(visited)

    while node != -1:
        order_parameter, visited_i = G.calculate_AF_order_parameter(node, order_parameter, vertices, l_z)
        visited.append(visited_i)
        node = not_all_visited(visited)

    AF_order_parameter = order_parameter / len(vertices)

    print("order parameter =", AF_order_parameter)

    return AF_order_parameter


def calculate_AF_order_parameter_old(G, vertices, edges, vertices_sign, visited, l_z):
    order_parameter = 0
    previous = ''
    for component in vertices_sign:
        i = 0
        while component[i] == 0:
            i += 1
        if vertices[i][2] > l_z / 2:
            up = component[i]
            down = -1 * component[i]
        else:
            up = -1 * component[i]
            down = component[i]
        for vertex in range(len(component)):
            if component[vertex] != 0:
                if vertices[vertex][2] > l_z / 2:
                    order_parameter += up * component[vertex]
                else:
                    order_parameter += down * component[vertex]

    AF_order_parameter = order_parameter / len(vertices)

    print("order parameter =", AF_order_parameter)

    return AF_order_parameter
