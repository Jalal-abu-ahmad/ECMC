from post_process_2 import utils
import matplotlib.pyplot as plt


def connectivity_Bipartiteness_AFism(list_of_edges, points, boundaries, theta):
    vertices, edges = keep_within_boundaries_and_non_isolated(list_of_edges, points, boundaries, theta)

    check_connectivity(vertices, edges)
    check_bipartiteness(vertices, edges)
    calculate_AF_order_parameter(vertices, edges)

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


def keep_within_boundaries_and_non_isolated(list_of_edges, points, boundaries, theta):
    vertices = []
    edges = []

    first_rotation = utils.rotate_points_by_angle(points, -theta)

    for (u, v), color, in_circuit in list_of_edges:

        if not utils.out_of_boundaries(first_rotation[u], boundaries[0]):
            vertices.append(points[u])
            if not utils.out_of_boundaries(first_rotation[v], boundaries[1]):
                vertices.append(points[v])
                if not (in_circuit or color == 'red'):
                    edges.append([len(vertices)-2, len(vertices)-1])
        else:
            if not utils.out_of_boundaries(first_rotation[v], boundaries[1]):
                vertices.append(points[v])

    return vertices, edges


def check_connectivity(vertices, edges):
    pass


def check_bipartiteness(vertices, edges):
    pass


def calculate_AF_order_parameter(vertices, edges):
    pass
