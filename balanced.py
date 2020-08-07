import numpy as np
import random
import time
import timeit
import scipy.spatial
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

##
# Davis Arthur
# ORNL
# Performs classical balanced k-means clustering
# 6-29-2020
##

def balanced_kmeans(X, k):
    ''' Perform balanced k-means algorithm, returns centroids and assignments
    
    Args:
        X - input data
        k - number of clusters
    
    Returns:
        centroids: numpy array containing a centroid in each row
        assignments: list indicating each points cluster assignment
    '''
    N = np.shape(X)[0]
    C = init_centroids(X, k)
    assignments = np.zeros(N, dtype = np.int8)
    while True:
        newC, new_assignments = update_centroids(X, k, calc_weights(X, C))
        if np.array_equal(assignments, new_assignments):
            break
        C = newC
        assignments = new_assignments
    return C, assignments % k

#############
## Example ##
#############

def test1():
    X = np.array([[1, 2], [1, 4], [9, 5], [9, 6]])
    k = 2
    M, assignments = balanced_kmeans(X, k)
    print(M)
    print()
    print(assignments)

######################
## Helper functions ##
######################

def init_centroids(X, k):
    ''' Initialize the centroids at random
    
    Args:
        X: input data
        k: number of clusters
    
    Returns:
        centroids
    '''
    N = np.shape(X)[0]
    indexes = random.sample(range(N), k)
    centroids = np.array(X[indexes[0]])
    for i in range(k - 1):
        centroids = np.vstack((centroids, X[indexes[i + 1]]))
    return centroids

def calc_weights(X, C):
    ''' Calculate weights matrix used for Hungarian algorithm in assignment step
    
    Args:
        X: input data
        C: centroids
    
    Returns:
        weights
    '''
    N = np.shape(X)[0]
    k = np.shape(C)[0]
    D = np.square(scipy.spatial.distance_matrix(X, C))
    weights = np.kron(np.ones(N // k), D)
    if N % k > 0:
        weights = np.hstack((weights, D[:,range(N % k)]))
    return weights

def update_centroids(X, k, D):
    ''' Update the centroids
    
    Args: 
        X: input data
        D: weights matrix (based on distance between centroids and points)
        k: number of clusters
    
    Returns:
        updated centriods
    '''
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    C = np.zeros((k, d))
    assignments = np.array(linear_sum_assignment(D)[1])

    # sum of all points in a cluster
    for i in range(N):
        C[assignments[i] % k] += X[i]

    # divide by the number of points in that cluster
    num_full = N % k
    for i in range(k):
        if i < N % k:
            C[i] /= np.ceil(N / k)
        else:
            C[i] /= np.floor(N / k)
    return C, assignments

