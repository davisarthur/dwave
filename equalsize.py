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
            
# Generate qubo model
# X - training data
# k - number of clusters
def genModel(X, k):
    N = np.shape(X)[0]         # number of points
    D = genD(X)      # distance matrix
    D /= np.sum(find_middle(D, k))
    D *= 3.0 
    alpha = 1.0
    beta = 2.0
    print(D)
    F = alpha * genF(N, k)   # column penalty matrix
    G = beta * genG(N, k)   # row penalty matrix
    print(F)
    print(G)

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

# Get the middle n entries in an N x N array
# D - N x N numpy array
# k - number of clusters
def find_middle(D, k):
    N = np.shape(D)[0]
    n = N * (N // k - 1)
    sorted = np.sort(D.flatten())
    start = (N ** 2 - n) // 2
    end = start + n
    return sorted[start:end]

# Get the smallest nonzero entries in a nonnegative 2D numpy array
# D - 2D numpy array
# N - number of elements returned as a 1D numpy array
def find_small(D, N):
    small = []
    small_max = None
    for i in range(np.shape(D)[0]):
        for j in range(np.shape(D)[1]):
            if D[i][j] == 0:
                continue
            if small_max == None:
                small.append(D[i][j])
                small_max = D[i][j]
                continue
            if len(small) < N:
                small.append(D[i][j])
                if small_max < D[i][j]:
                    small_max = D[i][j]
                continue
            if D[i][j] < small_max:
                small.remove(small_max)
                small.append(D[i][j])
                small_max = max(small)
    return np.array(small)                

# Embed QUBO model on D-Wave hardware, returns sampler
# model - QUBO model to embed
def embed(model):
    return EmbeddingComposite(DWaveSampler(solver={'qpu': True}))

# Run QUBO problem on D-Wave hardware, return sample set
# sampler - D-Wave sampler being used to solve the problem
# model - QUBO model to embed
# num_reads - number of reads during annealing
def run_quantum(sampler, model, num_reads_in = 1000):
    return sampler.sample(model, num_reads = num_reads_in, auto_scale = True)

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

    assignments = np.array([-1] * N)
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

def test_small():
    D = np.array([[0, 4, 10, 9], [4, 0, 10, 13], [10, 10, 0, 1], [9, 13, 1, 0]])
    N = 4
    print(find_small(D, N))

def test_middle():
    D = np.array([[0, 10, 10, 1, 34, 50], [10, 0, 4, 5, 20, 29], [10, 4, 0, 5, 8, 13], [1, 5, 5, 0, 25, 32], \
        [34, 20, 8, 25, 0, 1], [50, 29, 13, 32, 1, 0]])
    N = np.shape(D)[0]
    k = 2
    n = N * (N // k - 1)
    print(find_middle(D, k))

if __name__ == "__main__":
    test_quantum()
