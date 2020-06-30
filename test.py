import numpy as np
import time
import dimod
import equalsize
import anysize
from datetime import datetime
from dwave.system import DWaveSampler, EmbeddingComposite
from scipy.cluster.vq import vq, kmeans2

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

# X - data to be printed
def printData(X):
    output = "("
    first = True
    for i in range(np.shape(X)[0]):
        if not first:
            output += ", " + str(X[i])
        else:
            output += str(X[i])
            first = False
    output += ")"
    return output

def test(N, k, r = 1.0, o = 0.1):

    # data file
    f = open("test.txt", "a")
    f.write("\n" + str(datetime.now()))    # denote date and time that test begins

    X = genClustered(N, k, r, o)
    f.write("\nData: " + printData(X)) 

    # get classical solution
    start = time.time()
    centroids_classical, assignments_classical = lloyd(X, k)
    end = time.time()
    f.write("\nLloyd's algorithm time elapsed: " + str(end - start))
    f.write("\nLloyd's algorithm solution: " + str(classical_solution))

    # generate QUBO model
    start = time.time()
    model = equalsize.genModel(X, k)
    end = time.time()
    f.write("\nQUBO Preprocessing time elapsed: " + str(end - start))

    # get simulated annealing solution
    start = time.time()
    sampleset_sim = dimod.SimulatedAnnealingSampler().sample(model)
    end = time.time()
    f.write("\nSimulated annealing time elapsed: " + str(end - start))

    # simulated annealing postprocessing
    start = time.time()
    assignments_sim = equalsize.getAssignments(sampleset_sim.first.sample, X)
    centroids_sim = equalsize.getCentroids(assignments_sim)
    end = time.time()
    f.write("\nSimulated postprocessing time elapsed: " + str(end - start))
    f.write("\nSimulated annealing solution: " + equalsize.printCentroids(centroids_sim))
    f.write("\nSimulated annealing assignments: " + equalsize.printAssignments(assignments_sim))

    # get quantum annealing solution
    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    sampleset_quantum = sampler_auto.sample(model, num_reads=1000)
    f.write("\nQuantum annealing time elapsed: " \
        + str(float(sampleset_quantum.info["timing"]["total_real_time"]) / (10 ** 6.0)))
    
    # quantum postprocessing
    start = time.time()
    assignments_quantum = equalsize.getAssignments(sampleset_quantum.first.sample, X)
    centroids_quantum = equalsize.getCentroids(assignments_quantum)
    end = time.time()
    f.write("\nQuantum postprocessing time elapsed: " + str(end - start))
    f.write("\nQuantum annealing solution: " + equalsize.printCentroids(centroids_quantum))
    f.write("\nQuantum annealing assignments: " \
        + equalsize.printAssignments(assignments_quantum) + "\n")
    f.close()

def test5(N, k):

    # data file
    f = open("test3.txt", "a")
    f.write("\n" + str(datetime.now()))    # denote date and time that test begins

    d = 2
    X = genRand(N, d)
    f.write("\nData: " + printData(X)) 

    # get classical solution
    start = time.time()
    classical_solution = classical(X, k)
    end = time.time()
    f.write("\nLloyd's algorithm time elapsed: " + str(end - start))
    f.write("\nLloyd's algorithm solution: " + str(classical_solution))

    # generate QUBO model
    start = time.time()
    model = equalsize.genModel(X, k)
    end = time.time()
    f.write("\nQUBO Preprocessing time elapsed: " + str(end - start))

    # get simulated annealing solution
    start = time.time()
    sampleset_sim = dimod.SimulatedAnnealingSampler().sample(model)
    end = time.time()
    f.write("\nSimulated annealing time elapsed: " + str(end - start))

    # simulated annealing postprocessing
    start = time.time()
    assignments_sim = equalsize.getAssignments(sampleset_sim.first.sample, X)
    centroids_sim = equalsize.getCentroids(assignments_sim)
    end = time.time()
    f.write("\nSimulated postprocessing time elapsed: " + str(end - start))
    f.write("\nSimulated annealing solution: " + equalsize.printCentroids(centroids_sim))
    f.write("\nSimulated annealing assignments: " + equalsize.printAssignments(assignments_sim))

    # get quantum annealing solution
    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    sampleset_quantum = sampler_auto.sample(model, num_reads=1000)
    f.write("\nQuantum annealing time elapsed: " \
        + str(float(sampleset_quantum.info["timing"]["total_real_time"]) / (10 ** 6.0)))
    
    # quantum postprocessing
    start = time.time()
    assignments_quantum = equalsize.getAssignments(sampleset_quantum.first.sample, X)
    centroids_quantum = equalsize.getCentroids(assignments_quantum)
    end = time.time()
    f.write("\nQuantum postprocessing time elapsed: " + str(end - start))
    f.write("\nQuantum annealing solution: " + equalsize.printCentroids(centroids_quantum))
    f.write("\nQuantum annealing assignments: " \
        + equalsize.printAssignments(assignments_quantum) + "\n")
    f.close()


# Lloyd's algorithm
# X - data set
# k - number of clusters
def lloyd(X, k):
    return kmeans2(X, k)

if __name__ == "__main__":
    test3(20, 2)
