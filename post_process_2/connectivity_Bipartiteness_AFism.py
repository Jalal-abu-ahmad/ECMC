from post_process_2 import utils
import matplotlib.pyplot as plt
from post_process_2 import Graph


def connectivity_Bipartiteness_AFism(list_of_edges, points, boundaries, theta):
    G = Graph.Graph()

    vertices, edges = keep_within_boundaries_and_non_isolated(G, list_of_edges, points, boundaries, theta)

    check_connectivity(G, vertices, edges)
    check_bipartiteness(G, vertices, edges)
    calculate_AF_order_parameter(G, vertices, edges)

    print("plotting edges")
    for (p1, p2) in edges:
        x1, y1 = vertices[p1]
        x2, y2 = vertices[p2]

        plt.plot([x1, x2], [y1, y2], color='grey', alpha=1)
        plt.plot(x1, y1, 'bo', markersize=5)
        plt.plot(x2, y2, 'bo', markersize=5)

    utils.plot_boundaries(boundaries, theta)
    plt.axis([130, 200, 360, 410])
    plt.gca().set_aspect('equal')
    plt.show()


def keep_within_boundaries_and_non_isolated(G, list_of_edges, points, boundaries, theta):
    vertices = []
    edges = []

    first_rotation = utils.rotate_points_by_angle(points, -theta)

    for (u, v), color, in_circuit in list_of_edges:

        if not (in_circuit or color == 'red'):
            if not utils.out_of_boundaries(first_rotation[u], boundaries[0]):
                vertices.append(points[u])
                if not utils.out_of_boundaries(first_rotation[v], boundaries[1]):
                    vertices.append(points[v])
                    edges.append([len(vertices)-2, len(vertices)-1])
                    G.addEdge(len(vertices)-2, len(vertices)-1)
            else:
                if not utils.out_of_boundaries(first_rotation[v], boundaries[1]):
                    vertices.append(points[v])

    return vertices, edges


def check_connectivity(G, vertices, edges):
    i = 0
    visited = []
    print(min(G.graph))
    visited[0] = G.BFS(0)
    node = not_all_visited(visited[i])

    while node != -1:
        i += 1
        visited[i] = G.BFS()
        node = not_all_visited(visited[i])

    print("Graph has", len(visited), "connected_componentes")

    return visited


def not_all_visited(visited):

    for node in range(len(visited)):
        if not visited[node]:
            return node
    return -1


def check_bipartiteness(G, vertices, edges):
    pass


def calculate_AF_order_parameter(G, vertices, edges):
    pass
