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
    info["N"] = int(N_k_str[0])
    info["k"] = int(N_k_str[1])
    info["d"] = int(N_k_str[2])

    # read in classical algorithm time
    info["time_classical"] = float(f.readline().split(":")[1])

    # read in QUBO preprocessing time
    info["time_preprocessing"] = float(f.readline().split(":")[1])

    # read in embedding/postprocessing time
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
        postprocessing_q.append(info["time_postprocessing"])
    time_c = np.array(time_c)
    preprocessing_q = np.array(preprocessing_q)
    embedding_q = np.array(embedding_q)
    postprocessing_q = np.array(postprocessing_q)
    dev = np.array([np.std(time_c), np.std(preprocessing_q), np.std(postprocessing_q)]) 
    avg = np.array([np.average(time_c), np.average(preprocessing_q), np.average(postprocessing_q)]) 
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
    all_info = read_range("test_iris.txt")
    scores_c = []
    scores_q = []
    for info in all_info:
        scores_c.append(metrics.adjusted_rand_score(info["target"], info["assignments_classical"]))
        scores_q.append(metrics.adjusted_rand_score(info["target"], info["assignments_quantum"]))
    dev = np.array([np.std(scores_q), np.std(scores_c)])
    avg = np.array([np.average(scores_q), np.average(scores_c)])
    print(avg)
    print(dev)

if __name__ == "__main__":
    rand_index_analysis()
