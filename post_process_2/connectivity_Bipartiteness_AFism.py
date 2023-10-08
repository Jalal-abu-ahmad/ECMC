from matplotlib import pyplot as plt

import Graph


def connectivity_Bipartiteness_AFism(list_of_edges, points_with_z, boundaries, theta, l_z):
    G = Graph.Graph()

    vertices, edges = keep_within_boundaries_and_non_isolated(G, list_of_edges, points_with_z, boundaries, theta)

    visited, no_of_connected_components = check_connectivity(G, vertices, edges)
    vertices_sign, bipartite = check_bipartiteness(G, vertices, edges, visited)
    AF_order_parameter, vertices_color = calculate_AF_order_parameter(G, vertices, edges, vertices_sign, visited, l_z)

    parameters = map(str, [AF_order_parameter, no_of_connected_components, bipartite])
    return parameters


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
    no_of_connected_components = 1
    visited = [G.BFS(0)]
    node = not_all_visited(visited)

    while node != -1:
        i += 1
        visited.append(G.BFS(node))
        if G.graph[node]:
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
        # i = 0
        # while component[i] == 0:
        #     i += 1
        # if vertices[i][2] > l_z / 2:
        #     up = component[i]
        #     down = -1 * component[i]
        # else:
        #     up = -1 * component[i]
        #     down = component[i]
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

    for i in range(len(vertices)):
        p_x = vertices[i][0]
        p_y = vertices[i][1]
        color = vertices_color[i]
        plt.plot(p_x, p_y, color, markersize=5)
    plt.show()

    return AF_order_parameter, vertices_color

#######################################################################################################################


def calculate_AF_order_parameter_old(G, vertices, edges, vertices_sign, visited, l_z):

    full_sign = vertices_sign[0]
    order_parameter = 0
    for i in range(len(visited)):
        full_sign = [full_sign[j] + vertices_sign[i][j] for j in range(len(vertices_sign[0]))]

    for [u, v] in edges:

        if vertices[u][2] > l_z / 2 and vertices[v][2] > l_z / 2 \
                or vertices[u][2] < l_z / 2 and vertices[v][2] < l_z / 2:
            if full_sign[u] != full_sign[v]:
                order_parameter += 0  # +1 and -1
            else:
                print("WTF?")
                order_parameter += 1
        else:
            if full_sign[u] != full_sign[v]:
                order_parameter += 1
            else:
                order_parameter += 0.5

    AF_order_parameter = order_parameter / len(edges)

    print("order parameter =", AF_order_parameter)

    return AF_order_parameter
