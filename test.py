import numpy as np
import dimod
import equalsize
from dwave.system import DWaveSampler, EmbeddingComposite

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

def test3(N, k, r = 10.0, o = 0.1):
    X = genClustered(N, k, r, o)
    model = equalsize.genModel(X, k)
        
    # get simulated solution
    sampleset_sim = dimod.SimulatedAnnealingSampler().sample(model)
    print("Simulated Annealing Solution:" + str(sampleset_sim.first.sample))
    assignments_sim = equalsize.getAssignments(sampleset_sim.first.sample, X)
    equalsize.printAssignements(assignments_sim)
    equalsize.printCentroids(equalsize.getCentroids(assignments_sim))

    # get quantum annealing solution
    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    sampleset_quantum = sampler_auto.sample(model, num_reads=1000)
    print("\nQuantum Anealing Solution:" + str(sampleset_quantum.first.sample))
    assignments_quantum = equalsize.getAssignments(sampleset_quantum.first.sample, X)
    equalsize.printAssignements(assignments_quantum)
    equalsize.printCentroids(equalsize.getCentroids(assignments_quantum))
 
if __name__ == "__main__":
    test3(32, 2)
