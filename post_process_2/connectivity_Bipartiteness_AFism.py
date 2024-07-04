from matplotlib import pyplot as plt

import Graph


def connectivity_Bipartiteness_AFism(list_of_edges, points_with_z, boundaries, theta, l_z):
    G = Graph.Graph()

    vertices, edges = keep_within_boundaries_and_non_isolated(G, list_of_edges, points_with_z, boundaries, theta)

    visited, no_of_connected_components = check_connectivity(G, vertices, edges)
    vertices_sign, bipartite = check_bipartiteness(G, vertices, edges, visited)
    AF_order_parameter, vertices_color = calculate_AF_order_parameter(G, vertices, edges, vertices_sign, visited, l_z)

    connectivity_parameters = map(str, [AF_order_parameter, no_of_connected_components, bipartite])
    return connectivity_parameters, AF_order_parameter


def keep_within_boundaries_and_non_isolated(G, list_of_edges, points, boundaries, theta):
    edges = []
    N = len(points)
    vertices = points

    for (u, v), color, in_circuit in list_of_edges:
        if not (in_circuit or color == 'salmon'):
            if u < N and v < N:
                G.addEdge(u, v)
                edges.append([u, v])

    return vertices, edges


def check_connectivity(G, vertices, edges):
    i = 0
    no_of_connected_components = 1
    visited, component = G.BFS(0)
    visited = [visited]
    node = not_all_visited(visited)

    while node != -1:
        i += 1
        visited_i, component = G.BFS(node)
        visited.append(visited_i)
        if component > 300:
            no_of_connected_components += 1
        node = not_all_visited(visited)

    print("Graph has", len(visited), "connected componentes")

    return visited, no_of_connected_components


def not_all_visited(visited):
    full_visited = visited[0]

    for i in range(len(visited)):
        full_visited = [full_visited[j] or visited[i][j] for j in range(len(visited[0]))]

    for node in range(len(full_visited)):
        if not full_visited[node]:
            return node
    return -1


def check_bipartiteness(G, vertices, edges, visited):
    bipartite = True
    non_compatible_0, sign_0, visited_0 = G.check_if_Bipartite_BFS(0)
    if non_compatible_0 != 0:
        bipartite = False
    visited = [visited_0]
    node = not_all_visited(visited)
    sign = [sign_0]
    print("non =", non_compatible_0)

    while node != -1:
        non_compatible_i, sign_i, visited_i = G.check_if_Bipartite_BFS(node)
        if non_compatible_i != 0:
            bipartite = False
        visited.append(visited_i)
        sign.append(sign_i)
        print("non =", non_compatible_i)
        node = not_all_visited(visited)

    return sign, bipartite


def calculate_AF_order_parameter(G, vertices, edges, vertices_sign, visited, l_z):
    vertices_color = [None] * len(vertices)
    order_parameter = 0
    if vertices[0][2] > l_z / 2:
        up = vertices_sign[0][0]
        down = -1 * vertices_sign[0][0]
    else:
        up = -1 * vertices_sign[0][0]
        down = vertices_sign[0][0]

    for component in vertices_sign:
        for vertex in range(len(component)):
            if component[vertex] != 0:
                if vertices[vertex][2] > l_z / 2:
                    vertex_order_parameter = up * component[vertex]
                else:
                    vertex_order_parameter = down * component[vertex]
                order_parameter += vertex_order_parameter

                if vertex_order_parameter == 1:
                    vertices_color[vertex] = 'go'
                else:
                    vertices_color[vertex] = 'ro'

    AF_order_parameter = order_parameter / len(vertices)

    print("order parameter =", AF_order_parameter)

    # for i in range(len(vertices)):
    #     p_x = vertices[i][0]
    #     p_y = vertices[i][1]
    #     color = vertices_color[i]
    #     plt.plot(p_x, p_y, color, markersize=5)

    return AF_order_parameter, vertices_color
