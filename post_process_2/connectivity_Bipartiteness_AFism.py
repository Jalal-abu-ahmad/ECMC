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

    print("plotting edges")
    for (p1, p2) in edges:
        x1, y1 = vertices[p1][1, 2]
        x2, y2 = vertices[p2][1, 2]

        plt.plot([x1, x2], [y1, y2], color='grey', alpha=1)
        plt.plot(x1, y1, 'bo', markersize=5)
        plt.plot(x2, y2, 'bo', markersize=5)

    utils.plot_boundaries(boundaries, theta)
    plt.axis([130, 200, 360, 410])
    plt.gca().set_aspect('equal')
    plt.show()


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
    visited = [[G.BFS(0)]]
    node = not_all_visited(visited[i])

    while node != -1:
        i += 1
        visited.append(G.BFS(node))
        node = not_all_visited(visited[i])

    print("Graph has", len(visited), "connected componentes")

    return visited


def not_all_visited(visited):

    for node in range(len(visited)):
        if not visited[node]:
            return node
    return -1


def check_bipartiteness(G, vertices, edges, visited):
    non_compatible, sign_0 = G.check_if_Bipartite_BFS(0)
    sign = [[sign_0]]
    print("non =", non_compatible)
    for i in range(1, len(visited)):
        node = not_all_visited(visited[i])
        if node != -1:
            sign.append(G.check_if_Bipartite_BFS(node)[1])

    return sign


def calculate_AF_order_parameter(G, vertices, edges, vertices_sign, visited, l_z):

    order_parameter = 0
    i = 0
    while vertices_sign[0][i] == 0:
        i += 1
    if vertices[i][2] > l_z / 2:
        up = vertices_sign[0][i]
        down = -1 * vertices_sign[0][i]
    else:
        up = -1 * vertices_sign[0][i]
        down = vertices_sign[0][i]

    for component in vertices_sign:
        for vertex in range(len(component)):
            if component[vertex] != 0:
                if vertices[vertex][2] > l_z/2:
                    order_parameter += up * component[vertex]
                else:
                    order_parameter += down * component[vertex]

    AF_order_parameter = order_parameter/len(vertices)

    print("order parameter =", AF_order_parameter)

    return AF_order_parameter

