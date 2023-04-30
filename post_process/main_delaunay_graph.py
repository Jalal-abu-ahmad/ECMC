import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def less_first(a, b):
    return [a, b] if a < b else [b, a]


def get_closest_vector_in_length(original_vec, list_of_vecs) -> float:
    minimum_dist = float('inf')
    for v in list_of_vecs:
        diff = np.linalg.norm(v - original_vec)
        minimum_dist = min(minimum_dist, diff)
    return minimum_dist


def is_diagonal(edge, perfect_lattice_vectors_only_diags, perfect_lattice_vectors_only_no_diags, points) -> bool:
    vec = points[edge[1]] - points[edge[0]]
    x = get_closest_vector_in_length(vec, perfect_lattice_vectors_only_diags)
    y = get_closest_vector_in_length(vec, perfect_lattice_vectors_only_no_diags)
    # print("x = ", x, ", y = ",  y, ", edge = ", edge, ", vec = ", vec)
    if x > y:
        return False
    else:
        return True


def delaunay2edges(tri, perfect_lattice_vectors_only_diags, perfect_lattice_vectors_only_no_diags, points):
    list_of_edges = []
    for triangle in tri.simplices:
        for e1, e2 in [[0, 1], [1, 2], [2, 0]]:  # for all edges of triangle
            edge = less_first(triangle[e1], triangle[e2])  # always lesser index first
            if not is_diagonal(edge, perfect_lattice_vectors_only_diags, perfect_lattice_vectors_only_no_diags, points):
                list_of_edges.append(edge)
    array_of_edges = np.unique(list_of_edges, axis=0)  # remove duplicates
    list_of_lengths = []
    for p1, p2 in array_of_edges:
        x1, y1 = tri.points[p1]
        x2, y2 = tri.points[p2]
        list_of_lengths.append((x1 - x2) ** 2 + (y1 - y2) ** 2)
    array_of_lengths = np.sqrt(np.array(list_of_lengths))
    return array_of_edges, array_of_lengths


def plot_graph(points, L, N):
    # Perform Delaunay triangulation
    tri = Delaunay(points)

    a = L/ (np.sqrt(N) - 1)
    a1, a2 = np.array([a, 0]), np.array([0, a])
    # perfect_lattice_vectors = [n * a1 + m * a2 for n in range(-3, 3) for m in range(-3, 3)]
    perfect_lattice_vectors_only_diags = list(filter(lambda item: item is not None,
                                                     [(n * a1 + m * a2 if n != 0 and m != 0 else None) for n in
                                                      range(-3, 4) for m in range(-3, 4)]))
    # print("perfect_lattice_vectors_only_diags = ", perfect_lattice_vectors_only_diags)
    perfect_lattice_vectors_only_no_diags = list(filter(lambda item: item is not None, [
        (n * a1 + m * a2 if (n == 0 or m == 0) and not (n == 0 and m == 0) else None) for n in range(-3, 4) for m in
        range(-3, 4)]))

    # print("perfect_lattice_vectors_only_no_diags = ", perfect_lattice_vectors_only_no_diags)

    edges, lengths = delaunay2edges(tri, perfect_lattice_vectors_only_diags, perfect_lattice_vectors_only_no_diags, points)

    print(len(edges))

    array_of_edges = np.unique(edges, axis=0)  # remove duplicates
    plt.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices, color='red')
    x_lines = []
    y_lines = []
    for p1, p2 in array_of_edges:
        x1, y1 = points[p1]
        x2, y2 = points[p2]
        plt.plot([x1, x2], [y1, y2], color='blue')

    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


def case1():
    # Generate a square lattice of points
    L = 100  # Size of the square lattice
    N = 400  # Number of points
    x = np.linspace(0, L, int(np.sqrt(N)))
    y = np.linspace(0, L, int(np.sqrt(N)))
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).T
    # print(points)

    # Add some random defects to the lattice
    noise = np.random.normal(0, 1.0, size=points.shape)
    points += noise

    plot_graph(points=points, L=L, N=N)


def case2():
    # Generate a square lattice of points
    L = 100  # Size of the square lattice
    N = 400  # Number of points
    x = np.linspace(0, L, int(np.sqrt(N)))
    y = np.linspace(0, L, int(np.sqrt(N)))
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).T
    # print(points)

    # remove some chunk
    for i in range(N//2, N//2 + int(np.sqrt(N))):
        points = np.delete(points, i, axis=0)

    # Add some random defects to the lattice
    noise = np.random.normal(0, 0.4, size=points.shape)
    points += noise

    plot_graph(points=points, L=L, N=N)


if __name__ == "__main__":
    case2()
