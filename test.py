import numpy as np
import dimod
import equalsize
import anysize
from dwave.system import DWaveSampler, EmbeddingComposite
from scipy.cluster.vq import vq, kmeans, whiten

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
        X[i] = [r * np.cos(delta * (i % k)), r * np.sin(delta * (i % k))]
    return X

def test3(N, k, r = 1.0, o = 0.01):
    X = genClustered(N, k, r, o)
    model = equalsize.genModel(X, k)

    # get classical solution
    print("Classical solution: ")
    classical(X, k)

    # get simulated annealing solution
    sampleset_sim = dimod.SimulatedAnnealingSampler().sample(model)
    print("\nSimulated Annealing Solution:" + str(sampleset_sim.first.sample))
    assignments_sim = equalsize.getAssignments(sampleset_sim.first.sample, X)
    #equalsize.printAssignements(assignments_sim)
    equalsize.printCentroids(equalsize.getCentroids(assignments_sim))

    # get quantum annealing solution
    # sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    # sampleset_quantum = sampler_auto.sample(model, num_reads=1000)
    # print("\nQuantum Anealing Solution:" + str(sampleset_quantum.first.sample))
    # assignments_quantum = equalsize.getAssignments(sampleset_quantum.first.sample, X)
    # equalsize.printAssignements(assignments_quantum)
    # equalsize.printCentroids(equalsize.getCentroids(assignments_quantum))

def test4(N, k, r = 1.0, o = 0.01):
    X = genClustered(N, k, r, o)
    d = np.shape(X)[1]
    p = np.transpose(np.array([-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0]))
    model = anysize.genModel(X, k, p)

    # get classical solution
    print("Classical solution: ")
    classical(X, k)

    # get simulated annealing solution
    sampleset_sim = dimod.SimulatedAnnealingSampler().sample(model)
    print("\nSimulated Annealing Solution:" + str(sampleset_sim.first.sample))
    centroids_sim = anysize.getCentroids(sampleset_sim.first.sample, p, d, k)
    print("\n" + str(centroids_sim))

    # get quantum annealing solution
    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    sampleset_quantum = sampler_auto.sample(model, num_reads=1000)
    print("\nQuantum Anealing Solution:" + str(sampleset_quantum.first.sample))
    centroids_quantum = anysize.getCentroids(sampleset_quantum.first.sample, p, d, k)
    print("\n" + str(centroids_quantum))

def classical(X, k):
    centroids = kmeans(whiten(X), k)[0]
    equalsize.printCentroids(centroids)

if __name__ == "__main__":
    test4(16, 2)
