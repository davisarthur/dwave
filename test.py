import numpy as np

##
# Davis Arthur
# ORNL
# Generate sample data for k-means clustering data
# 6-19-2020
##

# Generate random training data matrix
# N - number of points
# d - dimension of each point
def genRand(N, d):
    return np.random.random((N, d)) 

# Generate 2d data that is naturally partitioned along a circle
# N - number of points
# k - number of clusters
# r - radius of circle
# o - std deviation between two points within a cluster
def genClustered(N, k, r, o):
    d = 2
    delta = 2 * np.pi / k
    X = np.zeros((N, d))
    for i in range(k):
        # append cluster center
        X[i * N // k] = [r * np.cos(delta * i), r * np.sin(delta * i)]
        # add N // k elements to each cluster
        for j in range(N // k - 1):
            X[i * N // k + j + 1] = [r * np.cos(delta * i) + o * np.random.normal(), \
                r * np.sin(delta * i) + o * np.random.normal()]
    # split the remaining elements as equally as possible
    for i in range(N // k * k, N):
        X[i] = [r * np.cos(delta * (i % k)), r * np.sin(delta * (i % k)]
    return X