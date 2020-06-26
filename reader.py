import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

##
# Davis Arthur
# ORNL
# Series of functions used to analyze results of k-means clustering algorithms
# 6-24-2020
##

# Reader for test3 in test.py
def read3(filename):

    # dictionary for extracted information
    info = {}

    f = open(filename, "r")
    
    # prompt user to get the start time of data to be extracted
    begin = input("Start time: ")
    line = ""
    while not begin in line:
        line = f.readline()

    # read in data matrix X
    arrX = f.readline().split("(")[1].split(")")[0].split(", ")
    N = len(arrX)
    d = len(arrX[0][1:-1].split())
    X = np.zeros((N, d))
    for i in range(N):
        X[i] = np.array([float(j) for j in arrX[i][1:-1].split()])
    info["X"] = X

    # read in Lloyd's algorithm time
    info["time_classical"] = float(f.readline().split(":")[1])

    # read in Lloyds's algorithm centroids
    centroids_classical_arr = f.readline().split("Centroid ")[1:]
    k = len(centroids_classical_arr)
    centroids_classical = np.zeros((k, d))
    for i in range(k):
        centroids_classical[i] = np.array([float(j) for j in \
            centroids_classical_arr[i].split("[")[1].split("]")[0].split()])
    info["centroids_classical"] = centroids_classical

    # read in QUBO preprocessing time
    info["time_preprocessing"] = float(f.readline().split(":")[1])

    # read in simulated annealing/postprocessing time
    info["time_annealing_sim"] = float(f.readline().split(":")[1])
    info["time_postprocessing_sim"] = float(f.readline().split(":")[1])

    # read in simulated annealing centroids
    centroids_sim_arr = f.readline().split("Centroid ")[1:]
    centroids_sim = np.zeros((k, d))
    for i in range(k):
        centroids_sim[i] = np.array([float(j) for j in \
            centroids_sim_arr[i].split("[")[1].split("]")[0].split()])
    info["centroids_sim"] = centroids_sim

    # read in simulated annealing assignments
    f.readline()
    assignments_sim = []
    for _ in range(k):
        cluster_contents_arr = f.readline().split("(")[1].split(")")[0].split(", ")
        cluster = np.zeros((len(cluster_contents_arr), d))
        for j in range(len(cluster_contents_arr)):
            cluster[j] = np.array([float(l) for l in \
                cluster_contents_arr[j].split("[")[1].split("]")[0].split()])
        assignments_sim.append(cluster)
    info["assignments_sim"] = assignments_sim

    # read in quantum annealing/postprocessing time
    info["time_annealing_quantum"] = float(f.readline().split(":")[1])
    info["time_postprocessing_quantum"] = float(f.readline().split(":")[1])

    # read in quantum annealing centroids
    centroids_quantum_arr = f.readline().split("Centroid ")[1:]
    centroids_quantum = np.zeros((k, d))
    for i in range(k):
        centroids_quantum[i] = np.array([float(j) for j in \
            centroids_quantum_arr[i].split("[")[1].split("]")[0].split()])
    info["centroids_quantum"] = centroids_quantum
    
    # read in quantum annealing assignments
    f.readline()
    assignments_quantum = []
    for _ in range(k):
        cluster_contents_arr = f.readline().split("(")[1].split(")")[0].split(", ")
        cluster = np.zeros((len(cluster_contents_arr), d))
        for j in range(len(cluster_contents_arr)):
            cluster[j] = np.array([float(l) for l in \
                cluster_contents_arr[j].split("[")[1].split("]")[0].split()])
        assignments_quantum.append(cluster)
    info["assignments_quantum"] = assignments_quantum
    
    # close the file and return dictionary
    f.close()
    return info

# Only valid for 2 dimensional data
# X input data matrix
# cIn - color of data points
def plotData(X, cIn, patches, size = 10):
    patches.append(mpatches.Patch(color = cIn, label = "data points"))
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    if d != 2:
        print("Error: Data is not of dimension 2")
        return
    for i in range(N):
        plt.scatter(X[i][0], X[i][1], c = cIn, s = size)
    return patches

# Only valid for 2 dimensional centroids
# M - centroids
# cIn - color of centroids
def plotCentroids(M, cIn, patches, labelIn = "default label", size = 10):
    k = np.shape(M)[0]
    d = np.shape(M)[1]
    patches.append(mpatches.Patch(color = cIn, label = labelIn))
    if d != 2:
        print("Error: Data is not of dimension 2")
        return
    for i in range(k):
        plt.scatter(M[i][0], M[i][1], c = cIn, s = size)
    return patches

# Only valid for 2 dimensional data
# A - assignments
# colors - colors of each assignment
def plot_assignment(A, colors, patches, labels = None, size = 10):
    k = len(A)
    if labels == None:
        labels = []
        for i in range(k):
            labels.append("Cluster " + str(i))
    for i in range(k):
        patches.append(mpatches.Patch(color = colors[i], label = labels[i]))
        N = np.shape(A[i])[0]
        d = np.shape(A[i])[1]
        if d != 2:
            print("Error: Data is not of dimension 2")
            return
        for j in range(N):
            plt.scatter(A[i][j][0], A[i][j][1], c = colors[i], s = size)
    return patches

def compare_centroids():
    info = read3("test3.txt")
    k = np.shape(info["centroids_classical"])[0]
    N = np.shape(info["X"])[0]
    patches = []
    patches = plotData(info["X"], "c", patches)
    patches = plotCentroids(info["centroids_classical"], "g", patches, \
        "classical centroids", size = 40)
    patches = plotCentroids(info["centroids_quantum"], "m", patches, \
        "quantum centroids")
    patches = plotCentroids(info["centroids_sim"], "y", patches, \
        "simulated centroids")
    plt.title("Centroid Analysis (" + str(N) + " points, " + str(k) \
        + " clusters)")
    plt.legend(handles = patches)
    # plt.xlim(-1.5, 1.5)
    # plt.ylim(-1.5, 1.5)
    plt.show()

def compare_time():
    info = read3("test3.txt")
    k = np.shape(info["centroids_classical"])[0]
    N = np.shape(info["X"])[0]
    n = 3
    preprocessing = (0.0, info["time_preprocessing"], info["time_preprocessing"])
    annealing = (info["time_classical"], info["time_annealing_sim"], \
        info["time_annealing_quantum"])
    postprocessing = (0.0, info["time_postprocessing_sim"], \
        info["time_postprocessing_quantum"])
    ind = np.arange(n) # the x locations for the groups
    width = 0.35
    p1 = plt.bar(ind, preprocessing, width, color = "c")
    p2 = plt.bar(ind, postprocessing, width, bottom = np.array(preprocessing), \
        color = "y")
    p3 = plt.bar(ind, annealing, width, bottom = np.array(preprocessing) + \
        np.array(postprocessing), color = "m")
    plt.ylabel("Time (s)")
    plt.title("Time Analysis (" + str(N) + " points, " + str(k) \
        + " clusters)")
    plt.xticks(ind, ("Lloyd's algorithm", "Simulated Annealing", \
        "Quantum Annealing"))
    plt.legend((p1[0], p2[0], p3[0]), ("preprocessing", "postprocessing", \
        "algorithm/annealing"))
    plt.yscale("log")
    plt.ylim(info["time_classical"] / 2.0, info["time_annealing_sim"] * 2.0)
    plt.show()

def compare_assignments():
    info = read3("test3.txt")
    k = np.shape(info["centroids_classical"])[0]
    N = np.shape(info["X"])[0]

    patches = []
    patches = plotData(info["X"], "k", patches, size = 40)
    patches = plot_assignment(info["assignments_sim"], ["r", "g", "c", "y"], \
        patches)
    plt.title("Simulated Annealing Assignments (" + str(N) + " points, " + str(k) \
        + " clusters)")
    plt.legend(handles = patches)
    # plt.xlim(-1.5, 1.5)
    # plt.ylim(-1.5, 1.5)
    plt.show()

    patches = []
    patches = plotData(info["X"], "k", patches, size = 40)
    patches = plot_assignment(info["assignments_quantum"], ["r", "g", "c", "y"], \
        patches)
    plt.title("Quantum Annealing Assignments (" + str(N) + " points, " + str(k) \
        + " clusters)")
    plt.legend(handles = patches)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()

if __name__ == "__main__":
    compare_centroids()
    compare_time()
    compare_assignments()
