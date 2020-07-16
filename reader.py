import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics

##
# Davis Arthur
# ORNL
# Series of functions used to analyze results of k-means clustering algorithms
# 6-24-2020
##

# Read a range of entries in "test.txt"
def read_range(filename = "newtest.txt"):
    all_info = []
    f = open(filename, "r")

    # prompt user to get the start time of data to be extracted
    start = input("Start time: ")
    end = input("End time: ")
    line = ""
    while not start in line:
        line = f.readline()
    
    # read first entry
    all_info.append(read_entry(f))
    while True:
        f.readline()
        if end in f.readline():
            all_info.append(read_entry(f))
            break
        all_info.append(read_entry(f))
    
    # close the file and return dictionary
    f.close()
    return all_info
            
# Reader for an individual entry in "test.txt"
# f - file
def read_entry(f):

    # dictionary for extracted information
    info = {}

    # read data generation type
    gen_type = f.readline()
    if "Iris" in gen_type:
        info["target"] = read_assignments(f)

    # read in (N, k)
    N_k_str = f.readline().split(":")[-1].split("(")[-1].split(")")[0].split(",")
    info["N"] = float(N_k_str[0])
    info["k"] = float(N_k_str[1])

    # read in data matrix X
    f.readline()  # ignore label
    info["X"] = read_array(f)

    # read in classical algorithm time
    info["time_classical"] = float(f.readline().split(":")[1])

    # read in classical algorithm centroids
    f.readline()  # ignore label
    info["centroids_classical"] = read_array(f)

    # read in classical algorithm assignments
    info["assignments_classical"] = read_assignments(f)

    # read in classical silhouette distance
    info["silhouette_classical"] = float(f.readline().split(":")[1])

    # read in QUBO preprocessing time
    info["time_preprocessing"] = float(f.readline().split(":")[1])
    
    line = f.readline()
    sim = True
    if "Simulated" in line:
        # read in simulated annealing/postprocessing time
        info["time_annealing_sim"] = float(line.split(":")[1])
        info["time_postprocessing_sim"] = float(f.readline().split(":")[1])

        # read in simulated annealing centroids
        f.readline()    # ignore label
        info["centroids_sim"] = read_array(f)

        # read in simulated annealing assignments
        info["assignments_sim"] = read_assignments(f)

        # read in simulated annealing silhouette distance
        info["silhouette_sim"] = float(f.readline().split(":")[1])

    else:
        info["time_finding_embedding"] = float(line.split(":")[1])
        sim = False
    
    # read in quantum annealing/postprocessing time
    if sim:
        info["time_finding_embedding"] = float(f.readline().split(":")[1])
    info["time_embedding"] = float(f.readline().split(":")[1])
    info["num_physical_variables"] = float(f.readline().split(":")[1])
    info["time_annealing_quantum"] = float(f.readline().split(":")[1])
    info["time_postprocessing_quantum"] = float(f.readline().split(":")[1])

    # read in quantum annealing centroids
    f.readline()    # ignore label
    info["centroids_quantum"] = read_array(f)
    
    # read in quantum annealing assignments
    info["assignments_quantum"] = read_assignments(f)

    # read in classical silhouette distance
    info["silhouette_quantum"] = float(f.readline().split(":")[1])
    
    return info

# Read a range of entries in "test_time.txt"
def read_time_range(filename = "test_time.txt"):
    all_info = []
    f = open(filename, "r")

    # prompt user to get the start time of data to be extracted
    start = input("Start time: ")
    end = input("End time: ")
    line = ""
    while not start in line:
        line = f.readline()
    
    # read first entry
    all_info.append(read_time_entry(f))
    while True:
        f.readline()
        if end in f.readline():
            all_info.append(read_time_entry(f))
            break
        all_info.append(read_time_entry(f))
    
    # close the file and return dictionary
    f.close()
    return all_info

# Reader for an individual entry in "test_time.txt"
# f - file
def read_time_entry(f):
    # dictionary for extracted information
    info = {}

    # read in (N, k)
    N_k_str = f.readline().split(":")[-1].split("(")[-1].split(")")[0].split(",")
    info["N"] = float(N_k_str[0])
    info["k"] = float(N_k_str[1])

    # read in classical algorithm time
    info["time_classical"] = float(f.readline().split(":")[1])

    # read in QUBO preprocessing time
    info["time_preprocessing"] = float(f.readline().split(":")[1])

    # read in embedding/postprocessing time
    info["time_embedding"] = float(f.readline().split(":")[1])
    info["time_postprocessing"] = float(f.readline().split(":")[1])

    return info

# Reads and returns a numpy array from a file.
# File must be on the first line of the array
def read_array(f):
    X_array = []
    while True:
        line = f.readline()
        last = False
        if "]]" in line:
            last = True
        point_array_str = line.split("[")[-1].split("]")[0].split()
        point_array = []
        for point in point_array_str:
            point_array.append(float(point))
        X_array.append(point_array)
        if last:
            break
    return np.array(X_array)

# Reads an assignment array from a file.
# File must be on the line with the assignment
def read_assignments(f):
    assignment_array = []
    for a in f.readline().split(":")[1].split("[")[-1].split("]")[0].split():
        assignment_array.append(a)
    return np.array(assignment_array)

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

def silhouette_plot():
    data = [[0.843, 0.807, 0.798, 0.893, 0.901, 0.247], [0.843, 0.807, 0.798, 0.893, 0.901, 0.361], \
        [0.843, 0.517, -0.318, 0.719, -0.380, 0.098]]
    X = np.arange(3)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
    ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
    plt.ylabel("Average Silhouette")
    plt.title("Time Analysis (" + str(N) + " points, " + str(k) \
        + " clusters)")
    ind = np.arange(n)
    plt.xticks(ind, ("(6, 3)", "(10, 2)", "(12, 2)", "(9, 3)", "(14, 2)", "(8, 4)"))
    ax.set_yticks(np.arange(-11, 11, 0.1))
    plt.show()

def silhouette_analysis():
    all_info = read_range()
    silhouettes_c = []
    silhouettes_q = []
    physical_variables = []
    for info in all_info:
        silhouettes_c.append(info["silhouette_classical"])
        silhouettes_q.append(info["silhouette_quantum"])
        physical_variables.append(info["num_physical_variables"])
    silhouettes_c = np.array(silhouettes_c)
    silhouettes_q = np.array(silhouettes_q)
    physical_variables = np.array(physical_variables)
    dev = np.array([np.std(silhouettes_c), np.std(silhouettes_q), np.std(physical_variables)]) 
    avg = np.array([np.average(silhouettes_c), np.average(silhouettes_q), np.average(physical_variables)])
    print(avg)
    print(dev)

def time_analysis():
    all_info = read_time_range()
    time_c = []
    preprocessing_q = []
    embedding_q = []
    postprocessing_q = []
    for info in all_info:
        time_c.append(info["time_classical"])
        preprocessing_q.append(info["time_preprocessing"])
        embedding_q.append(info["time_embedding"])
        postprocessing_q.append(info["time_postprocessing"])
    time_c = np.array(time_c)
    preprocessing_q = np.array(preprocessing_q)
    embedding_q = np.array(embedding_q)
    postprocessing_q = np.array(postprocessing_q)
    dev = np.array([np.std(time_c), np.std(preprocessing_q), np.std(embedding_q), \
        np.std(preprocessing_q + embedding_q), np.std(postprocessing_q)]) 
    avg = np.array([np.average(time_c), np.average(preprocessing_q), np.average(embedding_q), \
        np.average(preprocessing_q + embedding_q), np.average(postprocessing_q)]) 
    print(avg)
    print(dev)

def anneal_analysis():
    all_info = read_range()
    times = []
    for info in all_info:
        times.append(info["time_annealing_quantum"])
    times = np.array(times)
    dev = np.std(times)
    avg = np.average(times)
    print(avg)
    print(dev)

def rand_index_analysis():
    all_info = read_range()
    scores_c = []
    scores_q = []
    for info in all_info:
        scores_c.append(metrics.adjusted_rand_score(info["target"], info["assignments_classical"]))
        scores_q.append(metrics.adjusted_rand_score(info["target"], info["assignments_quantum"]))
    dev = np.array(np.std(scores_q), np.std(scores_c))
    avg = np.array(np.average(scores_q), np.average(scores_c))
    print(avg)
    print(dev)

if __name__ == "__main__":
    rand_index_analysis(filename = "test_iris.txt")
