import numpy as np
import random
import dimod
import dwavebinarycsp as csp
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding import embed_bqm, unembed_sampleset
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

# Return the sampler used for annealing
def set_sampler():
    return DWaveSampler(solver={'qpu': True})

# Find an embedding and embed on the hardware
# sampler - D-Wave sampler
# model - logical BQM model
# Returns embeddeding
def get_embedding(sampler, model):
    edge_list_model = []
    for key in model.adj.keys():
        for value in model.adj[key].keys():
            if (value, key) not in edge_list_model:
                edge_list_model.append((key, value))
    edge_list_sampler = []
    for key in sampler.adjacency.keys():
        for value in sampler.adjacency[key]:
            if (value, key) not in edge_list_sampler:
                edge_list_sampler.append((key, value))
    return find_embedding(edge_list_model, edge_list_sampler)

def embed(sampler, model, embedding):
    return embed_bqm(model, embedding, sampler.adjacency)

# Run the problem on D-Wave hardware
# sampler - D-Wave sampler being used to solve the problem
# embedded_model - QUBO model to embed
# num_reads - number of reads during annealing
def run_quantum(sampler, embedded_model, num_reads_in = 100):
    return sampler.sample(embedded_model, num_reads = num_reads_in, auto_scale = True)

# Run QUBO problem using D-Wave's simulated annealing, returns sample set
# model - QUBO model to embed
def run_sim(model):
    return dimod.SimulatedAnnealingSampler().sample(model)

# Return centroids and assignments
# X - input data
# embedded_solution - embedded solution set produced by annealing
# embedding - embedding used to convert from logical to embedded model
# model - logical BQM model
def postprocess(X, embedded_solution_set, embedding, model):
    sample_set = unembed_sampleset(embedded_solution_set, embedding, model)
    solution = sample_set.first.sample
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    k = len(solution) // N
    assignments = np.array([0] * N)
    M = np.zeros((k, d))
    cluster_sizes = np.zeros(k)
    for i in range(N):
        for j in range(k):
            if solution[i + j * N] == 1:
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
    sampler = set_sampler()
    embedding = get_embedding(sampler, model)
    embedded_model = embed(sampler, model, embedding)
    print("Number of physical variables: " + str(len(embedded_model.variables)))
    embedded_solution_set = run_quantum(sampler, embedded_model)
    M, assignments = postprocess(X, embedded_solution_set, embedding, model)
    print("Centroids: ")
    print(M)
    print("Assignments: " + str(assignments))

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
    test_quantum()