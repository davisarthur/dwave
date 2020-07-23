import numpy as np
import scipy.spatial.distance
import random
import dimod
import embedder
import time
import test
from datetime import datetime
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

# Generate F matrix
# N - number of points
# k - number of clusters
def genF(N, k):
    return np.ones((N, N)) - 2 * N / k * np.identity(N)

# Generate G matrix
# N - number of points
# k - number of clusters
def genG(N, k):
    return np.ones((k, k)) - 2 * np.identity(k)

# Generate D matrix
# X - training data
def genD(X):
    D = scipy.spatial.distance.pdist(X, 'sqeuclidean')
    D /= np.amax(D)
    return scipy.spatial.distance.squareform(D)

# Generate Q matrix
# N - number of points
# k - number of clusters
def genQ(N, k):
    Q = np.zeros((N * k, N * k))
    for i in range(N * k):
        Q[i][N * (i % k) + i // k] = 1.0
    return Q    

def rowpenalty(N, k):
    return np.kron(np.ones(k) - 2 * np.identity(k), np.identity(N))

# Generate qubo model
# X - training data
# k - number of clusters
def genA(X, k, alpha = None, beta = None):
    N = np.shape(X)[0]      # number of points
    if alpha == None:
        alpha = 1.0 / (2.0 * N / k - 1.0)
    if beta == None:
        beta = 1.0
    D = genD(X)             # distance matrix
    F = genF(N, k)          # column penalty matrix
    return np.kron(np.identity(k), D + alpha * F) + rowpenalty(N, k)

# Generate QUBO model
# X - input data
# k - number of clusters
def genModel(X, k, alpha = None, beta = None):
    N = np.shape(X)[0]
    return dimod.as_bqm(genA(X, k, alpha = None, beta = None), dimod.BINARY)

# Returns D-Wave sampler being used for annealing
def set_sampler():
    return DWaveSampler(solver={'qpu': True})

# Find a possible embedding for the hardware
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

# Embeds QUBO on the hardware
# sampler - D-Wave sampler being used
# model - QUBO model as BQM
# embedding - embedding returned from get_embedding
# returns embedded_model used in run_quantum
def embed(sampler, model, embedding):
    return embed_bqm(model, embedding, sampler.adjacency)

# Run the problem on D-Wave hardware
# sampler - D-Wave sampler being used to solve the problem
# embedded_model - QUBO model to embed
# num_reads - number of reads during annealing
def run_quantum(sampler, embedded_model, num_reads_in = 100):
    return sampler.sample(embedded_model, num_reads = num_reads_in, auto_scale = True)

# Run QUBO problem using D-Wave's simulated annealing, returns sample set
# model - BQM model to embed
def run_sim(model):
    return dimod.SimulatedAnnealingSampler().sample(model)

# Return centroids and assignments from binary solution of embedded model
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

# Return centroids and assignments from binary solution to logical model
# X - input data
# solution
def postprocess2(X, solution):
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

# Example using D-Wave's quantum annealing
# Note: Embedding is done without the use of D-Wave composite
def test_quantum():
    X = np.array([[1, 2], [1, 3], [9, 5], [9, 6]])  # input data
    N = 4
    k = 2
    model = genModel(X, k)  # returns BQM model (not yet embedded)
    sampler = set_sampler()  # sets the D-Wave sampler 
    embedding = get_embedding(sampler, model)   # finds an embedding on the smapler
    embedded_model = embed(sampler, model, embedding)   # embed on the D-Wave hardware
    print("Number of qubits used: " + str(len(embedded_model.variables))) 
    embedded_solution_set = run_quantum(sampler, embedded_model)    # run on the D-Wave hardware
    M, assignments = postprocess(X, embedded_solution_set, embedding, model)    # postprocess the solution
    print("Centroids: ")
    print(M)
    print("Assignments: " + str(assignments))

# Example using D-Wave's embedding composite (D-Wave does embedding for you)
def test_quantum2():
    X = np.array([[1, 2], [1, 3], [9, 5], [9, 6]])  # input data
    k = 2
    model = genModel(X, k)    # generate BQM model (not yet embedded)
    sampler = EmbeddingComposite(DWaveSampler())    # sets D-Wave's sampler, embedding is done automatically
    solution_set = sampler.sample(model, num_reads=100)    # run on the D-wave hardware
    M, assignments = postprocess2(X, solution_set.first.sample)    # postprocess the solution
    print("Centroids: ")
    print(M)
    print("Assignments: " + str(assignments))

# Example using D-Wave's simulated quantum annealing
def test_sim():
    X = np.array([[1, 2], [1, 4], [9, 5], [9, 6]])
    k = 2
    model = genModel(X, k)      # generate BQM model (not yet embedded)
    sample_set = run_sim(model)     # run on simulated solver
    M, assignments = postprocess2(X, sample_set.first.sample)    # postprocess the solution
    print("Centroids: ")
    print(M)
    print("Assignments: " + str(assignments))

# Example using Pras' embedding algorithm
def test_quantum3():
    X = np.array([[1, 2], [1, 4], [9, 5], [9, 6]])  # input data
    N = 4
    k = 2
    A = genA(X, k)
    sampler = set_sampler()  # sets the D-Wave sampler
    embedding_dict, embeddings, qubitfootprint = embedder.embedQubo(A, np.zeros(N * k))
    embedded_model = dimod.as_bqm(embedding_dict, dimod.BINARY)
    print("Number of qubits used: " + str(qubitfootprint)) 
    embedded_solution_set = run_quantum(sampler, embedded_model)    # run on the D-Wave hardware
    print(embedder.postProcessing(embedded_solution_set, embeddings, A)[1][0])
    M, assignments = postprocess2(X, embedder.postProcessing(embedded_solution_set, embeddings, A)[1][0])    # postprocess the solution
    print("Centroids: ")
    print(M)
    print("Assignments: " + str(assignments))

def test_embed_time(maxN, k, d):
    f = open("embedding_time.txt", "a")
    for N in range(k, maxN + 1):
        for _ in range(50):
            f.write(str(datetime.now()))
            X = test.gen_data(N, k, d)[0]
            f.write("\n(N, k, d): (" + str(N) + ", " + str(k) + ", " + str(d) + ")")
            f.write("\nNumber of variables: " + str(N * k))
            A = genA(X, k)
            b = np.zeros(N * k)
            start = time.time()
            embedding_dict, embeddings, qubitfootprint = embedder.embedQubo(A, b)
            end = time.time()
            f.write("\nTime to find embedding: " + str(end - start))
            f.write("\nQubit footprint: " + str(qubitfootprint) + "\n\n")
    embedding_dict, embeddings, qubitfootprint = embedder.embedQubo(A, np.zeros(N * k))

if __name__ == "__main__":
    test_embed_time(16, 4, 2)