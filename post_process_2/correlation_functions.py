import math
import numpy as np
from matplotlib import pyplot as plt
from post_process_2 import utils
from collections import Counter
from scipy.spatial import distance_matrix


def create_distance_histogram(points, boundaries, a):
    print("calculating correlations")
    N = len(points)
    lx, ly = boundaries[0], boundaries[1]
    # r_vec = []
    # dist = []
    k = [2*np.pi/a, 0]
    bins = create_bins(0, np.sqrt(lx ** 2 + ly ** 2) / 2, 0.1)
    r_axis = r_axis_calc(bins)
    pos_corr = [0]*len(r_axis)
    distances = distance_matrix(points, points)
    for i in range(N):
        print(i)
        for j in range(N):
            if i == j:
                continue
            else:
                r_vec = utils.cyclic_vec(boundaries, points[i], points[j])
                dist = distances[i][j]
                pair_correlation = np.cos(np.dot(k, r_vec))/(np.pi*dist)
                fill_histogram(dist, pair_correlation, bins, pos_corr)
    plt.plot(r_axis, pos_corr)


def create_bins(lower_bound, upper_bound, width):
    """ create_bins returns an equal-width (distance) partitioning.
        It returns an ascending list of tuples, representing the intervals.
        A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0
        and i < quantity, satisfies the following conditions:
            (1) bins[i][0] + width == bins[i][1]
            (2) bins[i-1][0] + width == bins[i][0] and
                bins[i-1][1] + width == bins[i][1]
    """
    no_of_bins = int(((upper_bound - lower_bound)/width) + 1)
    bins = []
    for i in range(no_of_bins):
        bins.append((i*width, (i+1)*width))

    return bins


def find_bin(value, bins):
    """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
        binning returns the smallest index i of bins so that
        bin[i][0] <= value < bin[i][1]
    """

    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1


def fill_histogram(indx_value, value, bins, hist):

    bin_index = find_bin(indx_value, bins)
    # print(value, bin_index, bins[bin_index])
    hist[bin_index] += value


def r_axis_calc(bins):
    r = [0]*len(bins)
    for i in range(len(bins)):
        r[i] = (bins[i][0] + bins[i][1])/2
    return r


def positional_correlation_calculation(distance_histogram, dist, r_vec, k, r_axis):
    pos_corr = [0]*len(dist)
    for i in range(len(distance_histogram)):
        pos_corr[i] = np.cos(np.dot(k, r_vec[i]))/(np.pi*dist[i])
    plt.plot(r_axis,pos_corr)


if __name__ == "__main__":

    bins = create_bins(0, 200, 0.1)
    hist = []
    values = [57.99,58,58.01,23,5,13,40,102,190, 56]
    for value in values:
        fill_histogram(value, bins, hist)
    print(len(bins))
    frequencies = Counter(hist)
    for index in frequencies:
        print(bins[index])
    print(frequencies)