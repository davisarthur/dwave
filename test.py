import numpy as np
import time
import dimod
import equalsize
import anysize
import balanced
from datetime import datetime
from dwave.system import DWaveSampler, EmbeddingComposite
from scipy.cluster.vq import vq, kmeans2
from sklearn import datasets

##
# Davis Arthur
# ORNL
# Generate sample data for k-means clustering data
# 6-19-2020
##

# Array specifying the number of points in each cluster
# N - Number of points
# k - Number of clusters
def cluster_sizes(N, k):
    cluster_sizes = []
    for i in range(N % k):
        cluster_sizes.append(N // k + 1)
    for j in range(k - N % k):
        cluster_sizes.append(N // k)
    return np.array(cluster_sizes)

# Generate data matrix and assignment array
# N - Number of points
# k - Number of clusters
# d - dimension of each data point
# sigma - standard deviation of each cluster
# max - max of a given coordinate
def genData(N, k, d, sigma = 1.0, max = 10.0):
    centersIn = np.zeros((k, d))
    for i in range(k):
        while True:
            valid = True
            new_center = 2.0 * max * (np.random.rand(d) - np.ones(d) / 2.0)
            for j in range(i + 1):
                l = np.linalg.norm(new_center - centersIn[j])
                if l < 3.0 * sigma:
                    valid = False
                    break
            if valid:
                centersIn[i] = new_center
                break
    return datasets.make_blobs(n_samples = cluster_sizes(N, k), n_features = d, \
        centers = centersIn, cluster_std = sigma, center_box = (-max, max))

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
    f.write(str(datetime.now()))    # denote date and time that test begins

    X = genClustered(N, k, r, o)
    f.write("\nData: \n" + str(X)) 

    # get classical solution
    start = time.time()
    centroids_classical, assignments_classical = balanced.balanced_kmeans(X, k)
    end = time.time()
    f.write("\nClassical algorithm time elapsed: " + str(end - start))
    f.write("\nClassical algorithm centroids:\n" + str(centroids_classical))
    f.write("\nClassical algorithm assignments: " + str(assignments_classical))

    # generate QUBO model
    start = time.time()
    model = equalsize.genModel(X, k)
    end = time.time()
    f.write("\nQUBO Preprocessing time elapsed: " + str(end - start))

    # get simulated annealing solution
    start = time.time()
    sampleset_sim = equalsize.run_sim(model)
    end = time.time()
    f.write("\nSimulated annealing time elapsed: " + str(end - start))

    # simulated annealing postprocessing
    start = time.time()
    centroids_sim, assignments_sim = equalsize.postprocess(X, sampleset_sim.first.sample)
    end = time.time()
    f.write("\nSimulated postprocessing time elapsed: " + str(end - start))
    f.write("\nSimulated annealing centroids:\n" + str(centroids_sim))
    f.write("\nSimulated annealing assignments: " + str(assignments_sim))

    # embed on the D-Wave
    start = time.time()
    sampler_quantum = equalsize.embed(model)
    end = time.time()
    f.write("\nQuantum embedding time elapsed: " + str(end - start))

    # get quantum annealing solution
    sampleset_quantum = equalsize.run_quantum(sampler_quantum, model)
    f.write("\nQuantum annealing time elapsed: " \
        + str(float(sampleset_quantum.info["timing"]["total_real_time"]) / (10 ** 6.0)))
    
    # quantum postprocessing
    start = time.time()
    centroids_quantum, assignments_quantum = equalsize.postprocess(X, sampleset_quantum.first.sample)
    end = time.time()
    f.write("\nQuantum postprocessing time elapsed: " + str(end - start))
    f.write("\nQuantum annealing centroids:\n" + str(centroids_quantum))
    f.write("\nQuantum annealing assignments: " + str(assignments_quantum) + "\n\n")
    f.close()

# Lloyd's algorithm
# X - data set
# k - number of clusters
def lloyd(X, k):
    return kmeans2(X, k)

if __name__ == "__main__":
    test(9, 3)
