import numpy as np
import random
import dimod
import dwavebinarycsp as csp
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
            D[i][j] = np.square(np.linalg.norm(X[i] - X[j]))
    return D
 
# Generate valid row configurations of W
# W - binary array (N x k)
def validRow(W):
    k = np.shape(W)[1]
    configs = []
    for i in range(k):
        config = []
        for j in range(k):
            if i == j:
                config.append(1)
            else:
                config.append(0)
        config = tuple(config)
        configs.append(config)
    return configs
            
# Generate qubo model
# X - training data
# k - number of clusters
def genModel(X, k):
    N = np.shape(X)[0]         # number of points
    D = genD(X)                # distance matrix
    D /= np.amax(D)            # normalized distance matrix
    D /= k * 2.0
    F = 2.0 * genF(N, k) / k     # column penalty matrix
    G = 2.0 * genG(N, k) / N         # row penalty matrix

    # create array of binary variable labels
    W = []
    for i in range(N):
        row = []
        for j in range(k):
            row.append("w" + str(i) + "_" + str(j))
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

# Embed QUBO model on D-Wave hardware, returns sampler
# model - QUBO model to embed
def embed(model):
    return EmbeddingComposite(DWaveSampler(solver={'qpu': True}))

# Run QUBO problem on D-Wave hardware, return sample set
# sampler - D-Wave sampler being used to solve the problem
# model - QUBO model to embed
# num_reads - number of reads during annealing
def run_quantum(sampler, model, num_reads_in = 100):
    return sampler.sample(model, num_reads = num_reads_in)

# Run QUBO problem using D-Wave's simulated annealing, returns sample set
# model - QUBO model to embed
def run_sim(model):
    return dimod.SimulatedAnnealingSampler().sample(model)

# Postprocessing of solution from annealing
# Note: if a point is assigned to more than one cluster, it will belong to the lower cluster
def postprocess(X, w):
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    k = len(w) // N

    assignments = np.array([0] * N)
    M = np.zeros((k, d))
    cluster_sizes = np.zeros(k)
    for i in range(N):
        for j in range(k):
            if w["w" + str(i) + "_" + str(j)] == 1:
                M[j] += X[i]
                cluster_sizes[j] += 1.0
                assignments[i] = j
                break

    for i in range(k):
        M[i] /= cluster_sizes[i]

    return M, assignments

def test_quantum():
    X = np.array([[1, 2], [1, 3], [1, 4], [9, 5], [9, 6]])
    k = 2
    model = genModel(X, k)
    sampler = embed(model)
    sample_set = run_quantum(sampler, model)
    M, assignments = postprocess(X, sample_set.first.sample)
    print(M)
    print()
    print(assignments)

def test_sim():
    X = np.array([[1, 2], [1, 3], [1, 4], [9, 5], [9, 6]])
    k = 2
    model = genModel(X, k)
    sample_set = run_sim(model)
    M, assignments = postprocess(X, sample_set.first.sample)
    print(M)
    print()
    print(assignments)

if __name__ == "__main__":
    test_sim()