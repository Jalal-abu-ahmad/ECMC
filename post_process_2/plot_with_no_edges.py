import utils
import numpy as np


def demo():
    # Generate a square lattice of points
    L = 100  # Size of the square lattice
    N = 10000  # Number of points
    noise = 1.0
    x = np.linspace(0, L, int(np.sqrt(N)))
    y = np.linspace(0, L, int(np.sqrt(N)))
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).T
    # print(points)

    # Add some random defects to the lattice
    noise = np.random.normal(0, noise, size=points.shape)
    points += noise

    utils.plot_points_with_no_edges(points=points)


def read_from_file():
    file_path = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/N=90000_h=0.8_rhoH=0.82_AF_triangle_ECMC/84426366"
    N = 90000
    rho_H = 0.82
    h=0.8
    L = utils.get_L(N=N, h=h, rho_H=rho_H)

    points = utils.read_points_from_file(file_path=file_path)
    assert points.shape == (N, 2)
    utils.plot_points_with_no_edges(points=points)


if __name__ == "__main__":
    read_from_file()
