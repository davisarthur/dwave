import numpy as np
import random
import math
from scipy.optimize import linear_sum_assignment

##
# Davis Arthur
# ORNL
# Balanced k-means clustering
# 6-29-2020
##

# Initialize the centroids using k random points from the input data
# X - input data
# k - number of clusters
def init_centroids(X, k):
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    C = np.zeros((k, d))
    centroid_indexes = []
    for i in range(k):
        while True:
            new_index = random.randint(0, N - 1)
            if not new_index in centroid_indexes:
                centroid_indexes.append(new_index)
                C[i] = X[new_index]
                break
    return C

# X - input data
# C - centroids
def calc_weights(X, C):
    N = np.shape(X)[0]
    k = np.shape(C)[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i][j] = np.linalg.norm(X[i] - C[j % k]) ** 2.0
    return D

# X - input data
# D - weights matrix (based on distance between centroids and points)
# k - number of clusters
def update_centroids(X, k, D):
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    C = np.zeros((k, d))
    assignments = linear_sum_assignment(D)[1]

    # sum of all points in a cluster
    for i in range(N):
        C[assignments[i] % k] += X[i]

    # divide by the number of points in that cluster
    num_full = N % k
    for i in range(k):
        if i < N % k:
            C[i] /= math.ceil(N / k)
        else:
            C[i] /= math.floor(N / k)
    return C

# X - input data
# k - number of clusters
def balanced_kmeans(X, k):
    C = init_centroids(X, k)
    while True:
        newC = update_centroids(X, k, calc_weights(X, C))
        if np.array_equal(newC, C):
            break
        C = newC
    return C

if __name__ == "__main__":
    test(100, 3)
