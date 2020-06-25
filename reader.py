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
def plotData(X, cIn, patches):
    patches.append(mpatches.Patch(color = cIn, label = "data points"))
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    if d != 2:
        print("Error: Data is not of dimension 2")
        return
    for i in range(N):
        plt.scatter(X[i][0], X[i][1], c = cIn)
    return N, patches

# Only valid for 2 dimensional centroids
# M - centroids
# cIn - color of centroids
def plotCentroids(M, cIn, patches, labelIn = "default label"):
    k = np.shape(M)[0]
    d = np.shape(M)[1]
    patches.append(mpatches.Patch(color = cIn, label = labelIn))
    if d != 2:
        print("Error: Data is not of dimension 2")
        return
    for i in range(k):
        plt.scatter(M[i][0], M[i][1], c = cIn)
    return k, patches

# Only valid for 2 dimensional data
# A - assignments
# colors - colors of each assignment
def plotAssignments(A):
    for i in range(len(A)):
        N = np.shape(A[i])[0]
        d = np.shape(A[i])[1]
        if d != 2:
            print("Error: Data is not of dimension 2")
            return
        for j in N:
            plt.scatter(A[i][j][0], A[i][j][1], c = colorsIn[i])

def compare_centroids():
    info = read3("test3.txt")
    patches = []
    N, patches = plotData(info["X"], "c", patches)
    k, patches = plotCentroids(info["centroids_classical"], "g", patches, "classical centroids")
    patches = plotCentroids(info["centroids_quantum"], "m", patches, "quantum centroids")[1]
    patches = plotCentroids(info["centroids_sim"], "y", patches, "simulated centroids")[1]
    plt.title("Centroid Performance (" + str(N) + "points, " + str(k) + "clusters)")
    plt.legend(handles = patches)
    plt.show()

if __name__ == "__main__":
    compare_centroids()
    