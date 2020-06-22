import numpy as np
import random
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite

##
# Davis Arthur
# ORNL
# Equal size k-means clustering problem as QUBO problem
# 6-17-2020
##

# Generate F
# N - number of points
# k - number of clusters
def genF(N, k):
    return np.ones((N, N)) - 2 * N / k * np.identity(N)

# Generate G
# N - number of points
# k - number of clusters
def genG(N, k):
    return np.ones((k, k)) - 2 * np.identity(k)

# Generate D
# X - training data
def genD(X):
    N = np.shape(X)[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i][j] = np.linalg.norm(X[i] - X[j])
    return D
 
# Generate qubo model
# X - training data
# k - number of clusters
# p1 - penalty multiplier for column constraints
# p2 - penalty multiplier for row constraints
def genModel(X, k, p1 = 10.0, p2 = 10.0):
    N = np.shape(X)[0]      # number of points
    D = genD(X)             # distance matrix
    F = p1 * genF(N, k)     # column penalty matrix
    G = p2 * genG(N, k)     # row penalty matrix

    # create array of binary variable labels
    W = []
    for i in range(N):
        row = []
        for j in range(k):
            row.append("w" + str(i) + str(j))
        W.append(row)

    linear = {}
    quadratic = {}

    # account for D term
    for l in range(k):
        for j in range(N):
            for i in range(N):
                if i == j and W[i][l] in linear:
                    linear[W[i][l]] = linear[W[i][l]] + D[i][j]
                elif (W[i][l], W[j][l]) in quadratic:
                    quadratic[(W[i][l], W[j][l])] = quadratic[(W[i][l], W[j][l])] + D[i][j]
                elif i == j:
                    linear[W[i][l]] = D[i][j]
                else:
                    quadratic[(W[i][l], W[j][l])] = D[i][j]

    # account for F term
    for l in range(k):
        for j in range(N):
            for i in range(N):
                if i == j and W[i][l] in linear:
                    linear[W[i][l]] = linear[W[i][l]] + F[i][j]
                elif (W[i][l], W[j][l]) in quadratic:
                    quadratic[(W[i][l], W[j][l])] = quadratic[(W[i][l], W[j][l])] + F[i][j]
                elif i == j:
                    linear[W[i][l]] = F[i][j]
                else:
                    quadratic[(W[i][l], W[j][l])] = F[i][j]

    # account for G term
    for l in range(N):
        for j in range(k):
            for i in range(k):
                if i == j and W[l][i] in linear:
                    linear[W[l][i]] = linear[W[l][i]] + G[i][j]
                elif (W[l][i], W[l][j]) in quadratic:
                    quadratic[(W[l][i], W[l][j])] = quadratic[(W[l][i], W[l][j])] + G[i][j]
                elif i == j:
                    linear[W[l][i]] = G[i][j]
                else:
                    quadratic[(W[l][i], W[l][j])] = G[i][j]

    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.Vartype.BINARY)
    
# Read final assignments of each point
# w - final binary vector (length = N * k)
# X - training data array (N x d)
def getAssignments(w, X):
    N = np.shape(X)[0]
    k = len(w) // N
    assignments = [[] for _ in range(k)]
    for i in range(N):
        for j in range(k):
            if w["w" + str(i) + str(j)] == 1:
                assignments[j].append(X[i])
    return assignments

# Print the assignments in the form "Cluster 1: (x1, x2, ..., xd)"
def printAssignements(assignments):
    i = 1
    for row in assignments:
        output = "Cluster " + str(i) + ": ("
        first = True
        for entry in row:
            if not first:
                output += ", "
            output += str(entry)
            first = False
        output += ")"
        print(output)
        i += 1

# Get centroids
def getCentroids(assignments):
    centroids = []
    d = len(assignments[0][0])
    for row in assignments:
        centroid = np.zeros(d)
        for entry in row:
            centroid += entry
        centroid /= len(row)
        centroids.append(centroid)
    return centroids

# Print centroids
def printCentroids(centroids):
    i = 1
    output = ""
    first = True
    for centroid in centroids:
        if not first:
            output += ", "
        output += "Centroid " + str(i) + ": " + str(centroid)
        first = False
        i += 1
    print(output)

# Test case for k = 2 clustering using quantum annealing
def test():
    X = np.array([[1, 2], [1, 3], [5, 1], [6, 2], [1, 1], [5, 2]])
    model = genModel(X, 2)
    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    sampleset = sampler_auto.sample(model, num_reads=1000)
    print("Quantum Anealing Solution: " + str(sampleset.first.sample))
    assignments = getAssignments(sampleset.first.sample, X)
    printAssignements(assignments)
    printCentroids(getCentroids(assignments))

# Exact solution to above test case for k = 2 clustering
def exact():
    X = np.array([[1, 2], [1, 3], [5, 1], [6, 2], [1, 1], [5, 2]])
    model = genModel(X, 2)
    sampleset = dimod.ExactSolver().sample(model)
    print("Exact Solution: " + str(sampleset.first.sample))
    assignments = getAssignments(sampleset.first.sample, X)
    printAssignements(assignments)
    printCentroids(getCentroids(assignments))
    
if __name__ == "__main__":
    exact()
    print()
    test()
