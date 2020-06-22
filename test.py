import numpy as np
import matplotlib.pyplot as plt
import random

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
        for j in range(N // k - 1):
            X[i * N // k + j + 1] = [r * np.cos(delta * i) + o * np.random.normal(), \
                r * np.sin(delta * i) + o * np.random.normal()]
    for i in range(N // k * k + 1, N):
        randCluster = random.randint(0, k - 1)
        X[i] = [r * np.cos(delta * i), r * np.sin(delta * i)]
    return X

def test():
    X = genClustered(100, 3, 10.0, 1.0)
    for i in range(X.shape[0]):
        plt.scatter(X[i][0], X[i][1], c = "g")
    plt.show()

if __name__ == "__main__":
    test()
