import numpy as np
import random
import dimod
import dwavebinarycsp as csp
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding import embed_bqm
from minorminer import find_embedding
from dimod.traversal import connected_components

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

def set_sampler():
    return DWaveSampler(solver={'qpu': True})

# Embed QUBO model on D-Wave hardware, returns sampler
def embed():
    return EmbeddingComposite(DWaveSampler(solver={'qpu': True}))

def embed2(sampler, model):
    graph = model.adj
    final_graph = dict()
    for key in graph.keys():
        for value in graph[key].keys():
            final_graph[key].add(value)
    print(final_graph)
    embedding = find_embedding(final_graph, sampler.adjacency)
    return sampler, embed_bqm(model, embedding, sampler.adjacency)

# Run QUBO problem on D-Wave hardware, return sample set
# sampler - D-Wave sampler being used to solve the problem
# model - QUBO model to embed
# num_reads - number of reads during annealing
def run_quantum(sampler, model, num_reads_in = 100):
    return sampler.sample(model, num_reads = num_reads_in, auto_scale = True)

def run_quantum2(sampler, model, num_reads_in = 100):
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
    X = np.array([[1, 2], [1, 3], [9, 5], [9, 6]])
    k = 2
    model = genModel(X, k)
    sampler = embed()
    sample_set = run_quantum(sampler, model)
    dwave.inspector.show(sample_set)
    M, assignments = postprocess(X, sample_set.first.sample)
    print(M)
    print()
    print(assignments)

def test_quantum2():
    X = np.array([[1, 2], [1, 3], [9, 5], [9, 6]])
    k = 2
    model = genModel(X, k)
    sampler = set_sampler()
    embedded_model = embed2(sampler, model)
    sample_set = run_quantum2(sampler, embedded_model)
    print(sample_set)
    dwave.inspector.show(sample_set)
    M, assignments = postprocess(X, sample_set.first.sample)
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


if __name__ == "__main__":
    test_quantum2()