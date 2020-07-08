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

# Generate Q
# N - number of points
# k - number of clusters
def genQ(N, k):
    Q = np.zeros((N * k, N * k))
    for i in range(N * k):
        Q[i][N * (i % k) + i // k] = 1.0
    return Q    

# Generate qubo model
# X - training data
# k - number of clusters
def genA(X, k, alpha, beta):
    N = np.shape(X)[0]      # number of points
    D = genD(X)             # distance matrix
    D /= np.amax(D)
    F = genF(N, k)          # column penalty matrix
    G = genG(N, k)          # row penalty matrix
    Q = genQ(N, k)
    return np.kron(np.identity(k), D + alpha * F) \
        + np.matmul(np.matmul(np.transpose(Q), np.kron(np.identity(N), beta * G)), Q)

# Generate QUBO model
# X - input data
# k - number of clusters
def genModel(X, k, alpha = None, beta = None):
    N = np.shape(X)[0]
    if alpha == None:
        alpha = 1.0 / (2.0 * N / k - 1.0)
    if beta == None:
        beta = 1.0
    return dimod.as_bqm(genA(X, k, alpha, beta), dimod.BINARY)

# Embed QUBO model on D-Wave hardware, returns sampler
# model - QUBO model to embed
def embed(model):
    return EmbeddingComposite(DWaveSampler(solver={'qpu': True}))

# Run QUBO problem on D-Wave hardware, return sample set
# sampler - D-Wave sampler being used to solve the problem
# model - QUBO model to embed
# num_reads - number of reads during annealing
def run_quantum(sampler, model, num_reads_in = 100):
    return sampler.sample(model, num_reads = num_reads_in, auto_scale = True)

# Run QUBO problem using D-Wave's simulated annealing, returns sample set
# model - QUBO model to embed
def run_sim(model):
    return dimod.SimulatedAnnealingSampler().sample(model)

def postprocess(X, w):
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    k = len(w) // N
    assignments = np.array([0] * N)
    M = np.zeros((k, d))
    cluster_sizes = np.zeros(k)
    for i in range(N):
        for j in range(k):
            if w[i + j * N] == 1:
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
    print(model.quadratic)
    model2 = genModel2(X, k)
    print(model2.quadratic)
    sampler = embed(model)
    sample_set = run_quantum(sampler, model)
    print(sample_set.first.sample)
    M, assignments = postprocess2(X, sample_set.first.sample)
    print(M)
    print()
    print(assignments)

def test_sim():
    X = np.array([[1, 2], [1, 3], [1, 4], [9, 5], [9, 6]])
    k = 2
    model = genModel(X, k)
    model2 = genModel2(X, k)
    print(model.quadratic)
    print()
    print(model2.quadratic)
    sample_set = run_sim(model2)
    print(sample_set.first.sample)
    
    M, assignments = postprocess2(X, sample_set.first.sample)
    print(M)
    print()
    print(assignments)

def test_middle():
    D = np.array([[0, 10, 10, 1, 34, 50], [10, 0, 4, 5, 20, 29], [10, 4, 0, 5, 8, 13], [1, 5, 5, 0, 25, 32], \
        [34, 20, 8, 25, 0, 1], [50, 29, 13, 32, 1, 0]])
    N = np.shape(D)[0]
    k = 2
    n = N * (N // k - 1)
    print(find_middle(D, k))

def test_Q():
    print(genQ(4, 2))

if __name__ == "__main__":
    test_sim()