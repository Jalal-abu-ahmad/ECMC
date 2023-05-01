import utils
import numpy as np


def demo():
    # Generate a square lattice of points
    L = 20  # Size of the square lattice
    N = 100  # Number of points
    noise = 0.1
    rotation_angle_degree = 10
    rotation_angle = np.deg2rad(rotation_angle_degree)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), np.sin(rotation_angle)],
        [-np.sin(rotation_angle), np.cos(rotation_angle)],
    ])
    x = np.linspace(0, L, int(np.sqrt(N)))
    y = np.linspace(0, L, int(np.sqrt(N)))
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).T
    rotated_points = points @ rotation_matrix
    assert rotated_points.shape == (N, 2)
    # print(points)

    # Add some random defects to the lattice
    noise = np.random.normal(0, noise, size=rotated_points.shape)
    rotated_points += noise

    utils.plot_points_with_no_edges(points=rotated_points)


def read_from_file():
    file_path = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.82_AF_triangle_ECMC/84426366"
    N = 90000
    rho_H = 0.82
    h=0.8
    L = utils.get_L(N=N, h=h, rho_H=rho_H)

    points = utils.read_points_from_file(file_path=file_path)
    assert points.shape == (N, 2)
    utils.plot_points_with_delaunay_edges_where_diagonals_are_removed(points=points, N=N, L=L)


if __name__ == "__main__":
    demo()
