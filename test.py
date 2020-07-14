import numpy as np
import time
import dimod
import equalsize
import anysize
import balanced
import random
from datetime import datetime
from dwave.system import DWaveSampler, EmbeddingComposite
from sklearn import datasets, metrics

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


def silhouette_analysis(X, assignments):
    return metrics.silhouette_score(X, assignments)

# Test using synthetic data. Results are written to "test.txt"
# N - Number of points
# k - Number of clusters
# d - dimension of each data point
# sigma - standard deviation of each cluster
# max - max of a given coordinate
def test(N, k, d = 2, sigma = 1.0, max = 10.0):

    # data file
    f = open("test.txt", "a")
    f.write(str(datetime.now()))    # denote date and time that test begins
    X = genData(N, k, d, sigma = 1.0, max = 10.0)[0]
    f.write("\n(N, k): " + "(" + str(N) + ", " + str(k) + ")")
    f.write("\nData: \n" + str(X)) 

    # get classical solution
    start = time.time()
    centroids_classical, assignments_classical = balanced.balanced_kmeans(X, k)
    end = time.time()
    f.write("\nClassical algorithm time elapsed: " + str(end - start))
    f.write("\nClassical algorithm centroids:\n" + str(centroids_classical))
    f.write("\nClassical algorithm assignments: " + str(assignments_classical))
    f.write("\nClassical silhouette distance: " + str(silhouette_analysis(X, assignments_classical)))

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
    f.write("\nSimulated annealing silhouette distance: " + str(silhouette_analysis(X, assignments_sim)))

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
    f.write("\nQuantum annealing assignments: " + str(assignments_quantum))
    f.write("\nQuantum annealing silhouette distance: " + str(silhouette_analysis(X, assignments_quantum)) + "\n\n")
    f.close()

# Generates a plausible synthetic binary solution to test the 
# time performance of postprocessing function
# N - Number of points
# k - Number of clusters
def synthetic_w(N, k):
    w = {}
    for i in range(N):
        one_index = random.randint(0, k - 1) 
        for j in range(k):
            if j == one_index:
                w[i + j * N] = 1
            else:
                w[i + j * N] = 0
    return w
        
def test_time(N, k, d = 2, sigma = 1.0, max = 10.0):
    # data file
    f = open("test_time3.txt", "a")
    f.write(str(datetime.now()))    # denote date and time that test begins

    X = genData(N, k, d, sigma = 1.0, max = 10.0)[0]
    f.write("\n(N, k): " + "(" + str(N) + ", " + str(k) + ")")

    # get classical solution
    start = time.time()
    centroids_classical, assignments_classical = balanced.balanced_kmeans(X, k)
    end = time.time()
    f.write("\nClassical algorithm time elapsed: " + str(end - start))

    # generate QUBO model
    start = time.time()
    model = equalsize.genModel(X, k)
    end = time.time()
    f.write("\nQUBO Preprocessing time elapsed: " + str(end - start))

    # embed on the D-Wave
    start = time.time()
    sampler_quantum = equalsize.embed(model)
    end = time.time()
    f.write("\nQuantum embedding time elapsed: " + str(end - start))

    # postprocess synthetic data
    w = synthetic_w(N, k)
    start = time.time()
    assignments_quantum = equalsize.postprocess(X, w)
    end = time.time()
    f.write("\nQuantum postprocessing time elapsed: " + str(end - start) + "\n\n")
    f.close()

def test_synthetic():
    print(synthetic_w(8, 2))

# Tests the QUBO model on a set of points from the iris dataset
# N - Number of data points
# k - Number of clusters
def test_iris(N, k):
    iris = datasets.load_iris()
    length = 150                    # total number of points in the dataset
    pp_cluster = 50                 # number of data points belonging to any given iris
    d = 4                           # iris dataset is of dimension 4
    data = iris["data"]             # all iris datapoints in a (150 x 4) numpy array
    full_target = iris["target"]    # all iris assignments in a list of length 150

    num_full = N % k        # number of clusters with maximum amount of entries
    available = [True] * length

    # build the data matrix and target list
    X = np.zeros((N, d))
    target = [-1] * N

    for i in range(k):
        for j in range(N // k):
            num = random.randint(0, pp_cluster - j - 1)
            count = 0
            for l in range(pp_cluster):
                if count == num:
                    X[i * N // k + j] = data[i * pp_cluster + l]
                    target[i * N // k + j] = full_target[i * pp_cluster + l]
                    print(target)
                    break
                if available[l]:
                    count += 1

    for i in range(num_full):
        num = random.randint(0, pp_cluster - N // k - 1)
        count = 0
        for l in range(pp_cluster):
            if count == num:
                X[N // k * k + i] = data[i * pp_cluster + l]   
                target[N // k * k + i] = full_target[i * pp_cluster + l]                 
                break
            if available[l]:
                count += 1
    return X, target

if __name__ == "__main__":
    X, target = test_iris(5, 3)
    # print(X)
    # print(target)

