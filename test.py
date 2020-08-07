import numpy as np
import time
import dimod
import equalsize
import anysize
import balanced
import embedder
import random
import itertools
import scipy.spatial.distance
from sklearn.cluster import KMeans
from datetime import datetime
from dwave.system import DWaveSampler, EmbeddingComposite
from sklearn import datasets, metrics

##
# Davis Arthur
# ORNL
# A number of performance tests for the classical algorithms and quantum algorithm
# 6-19-2020
##

def gen_data(N, k, d, d_informative = None, sep = 1.0):
    ''' Generate synthetic classification dataset
    
    Args: 
        N: Number of points
        k: Number of clusters
        d: dimension of each data point
        d_informative: number of features that are important to classification
        sep: factor used to increase or decrease seperation between clusters
    
    Returns: 
        X: training data as numpy array
        labels: class assignment of each point
    '''
    if d_informative == None:
        d_informative = d
    return datasets.make_classification(n_samples=N, n_features=d, \
        n_informative=d_informative, n_redundant=0, n_classes=k, \
        n_clusters_per_class=1, flip_y=0.01, class_sep=sep)

def gen_iris(N, k):
    ''' Generate data set from Iris dataset
    
    Args: 
        N: Number of points
        k: Number of clusters
    
    Returns: 
        X: training data as numpy array
        target: class assignment of each point
    '''
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
                    X[i * (N // k) + j] = data[i * pp_cluster + l]
                    target[i * (N // k) + j] = full_target[i * pp_cluster + l]
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

def test(N, k, d = 2, filename = "test.txt", data = "synthetic", sim = False, specs = None):
    ''' Test our quantum algorithm and both classical approaches. 
    Note: Results are written to "test.txt" by default
    
    Args:
        N: Number of points
        k: Number of clusters
        d: Dimension of each data point (only used if data = "synthetic")
        filename: File to write to
        data: How data is generated ("synthetic" (default) or "iris")
        sim: Run using simulated annealing (default = False)
    '''
    # data file
    f = open(filename, "a")
    f.write(str(datetime.now()))    # denote date and time that test begins

    if data == "synthetic":
        X, target = gen_data(N, k, d)
        f.write("\nSynthetic data")
        f.write("\nTarget: " + str(target))
    elif data == "iris":
        X, target = gen_iris(N, k)
        f.write("\nIris")
        f.write("\nTarget: " + str(target))
    else:
        print("Unsupported data generation technique.")
        return

    f.write("\n(N, k): " + "(" + str(N) + ", " + str(k) + ")")
    f.write("\nData: \n" + str(X)) 

    # get sklearn solution
    start = time.time()
    kmeans = KMeans(n_clusters=k).fit(X)
    end = time.time()
    centroids_sklearn = kmeans.cluster_centers_
    assignments_sklearn = kmeans.labels_
    f.write("\nSKlearn algorithm time elapsed: " + str(end - start))
    f.write("\nSKlearn algorithm centroids:\n" + str(centroids_sklearn))
    f.write("\nSKlearn algorithm assignments: " + str(assignments_sklearn))

    # get balanced solution
    start = time.time()
    centroids_balanced, assignments_balanced = balanced.balanced_kmeans(X, k)
    end = time.time()
    f.write("\nBalanced algorithm time elapsed: " + str(end - start))
    f.write("\nBalanced algorithm centroids:\n" + str(centroids_balanced))
    f.write("\nBalanced algorithm assignments: " + str(assignments_balanced))

    if not specs == "classical only":
        # generate QUBO model
        start = time.time()
        model = equalsize.genModel(X, k)
        end = time.time()
        f.write("\nQUBO Preprocessing time elapsed: " + str(end - start))

        if sim:
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
        sampler_quantum = equalsize.set_sampler()
        start = time.time()
        embedding = equalsize.get_embedding(sampler_quantum, model)
        end = time.time()
        f.write("\nFinding embedding time elapsed: " + str(end - start))
        
        # create BQM model from embedding
        start = time.time()
        embedded_model = equalsize.embed(sampler_quantum, model, embedding)
        end = time.time()
        f.write("\nTime to embed: " + str(end - start))

        # get quantum annealing solution
        embedded_solution_set = equalsize.run_quantum(sampler_quantum, embedded_model)
        f.write("\nNumber of Physical variables: " + str(len(embedded_model.variables)))
        f.write("\nQuantum annealing time elapsed: " \
            + str(float(embedded_solution_set.info["timing"]["total_real_time"]) / (10 ** 6.0)))
        
        # quantum postprocessing
        start = time.time()
        centroids_quantum, assignments_quantum = equalsize.postprocess(X, embedded_solution_set, embedding, model)
        end = time.time()
        f.write("\nQuantum postprocessing time elapsed: " + str(end - start))
        f.write("\nQuantum annealing centroids:\n" + str(centroids_quantum))
        f.write("\nQuantum annealing assignments: " + str(assignments_quantum) + "\n\n")
        f.close()
    else:
        f.write("\n\n")
        f.close()        

def test_time(N, k, d = 2, sigma = 1.0, specs = None):
    ''' Only test the time required for the three clustering approaches
    Used for examples too large for the D-Wave hardware

    Args:
        N: number of points
        k: number of clusters
        d: dimension of dataset
        sigma: std dev within a cluster
        specs: "classical only" if you don't want to run quantum approach
    '''
    # data file
    f = open("test_time2.txt", "a")
    f.write(str(datetime.now()))    # denote date and time that test begins

    X = gen_data(N, k, d)[0]
    f.write("\n(N, k, d): " + "(" + str(N) + ", " + str(k) + ", " + str(d) + ")")

    # get sklearn solution
    start = time.time()
    kmeans = KMeans(n_clusters=k).fit(X)
    end = time.time()
    f.write("\nSKlearn algorithm time elapsed: " + str(end - start))

    # get balanced algorithm solution
    start = time.time()
    classical_solution = balanced.balanced_kmeans(X, k)
    end = time.time()
    f.write("\nBalanced algorithm time elapsed: " + str(end - start))

    if not specs == "classical only":
        # generate QUBO model
        start = time.time()
        model = equalsize.genA(X, k)
        end = time.time()
        f.write("\nQUBO matrix formulation time elapsed: " + str(end - start))

        # postprocess synthetic data
        start = time.time()
        assignments_quantum = equalsize.postprocess2(X, kmeans.labels_)
        end = time.time()
        f.write("\nQuantum postprocessing time elapsed: " + str(end - start))
        f.write("\n\n")
        f.close()
    else:
        f.write("\n\n")
        f.close()

##############
## Examples ##
##############

def test_iris(N, k):
    X, target = gen_iris(N, k)
    model = equalsize.genModel(X, k)
    sampler = equalsize.embed()
    sample_set = equalsize.run_quantum(sampler, model)
    dwave.inspector.show(sample_set)
    M, assignments = equalsize.postprocess(X, sample_set.first.sample)
    print("Target: " + str(target))
    print("Assignments: " + str(assignments))
    
def test_synth(N, k, d = 2):
    X = gen_data(N, k, d)[0]
    model = equalsize.genModel(X, k)
    sampler = equalsize.embed()
    sample_set = equalsize.run_quantum(sampler, model)
    print(sample_set.info["timing"])
    print(sample_set.info.keys())
    # dwave.inspector.show(sample_set)
    M, assignments = equalsize.postprocess(X, sample_set.first.sample)

if __name__ == "__main__":
    all_configs = [(1024, 4, 32), (1024, 4, 64), (1024, 4, 128), (1024, 4, 256)]
    for i in range(len(all_configs)):
        for _ in range(50):
            test_time(all_configs[i][0], all_configs[i][1], all_configs[i][2])
