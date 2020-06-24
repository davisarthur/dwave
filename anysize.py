import numpy as np 
import dimod

##
# Davis Arthur
# ORNL
# Equal size k-means clustering problem as QUBO problem
# 6-23-2020
##

# Generate the precision matrix (d x (l * d))
# p - precision column vector (length l)
# d - dimension of each point
def genP(p, d):
    return np.kron(np.identity(d), p)

# Generate labels for binary variables in M. Each row in M is one of the k-means.
# k - number of clusters
# d - dimension of each data point
# l - length of precision vector
def genMeanLabels(k, d, l):
    M = []
    for i in range(k):
        m = []
        for j in range(d * l):
            m.append("M" + str(i) + "_" + str(j))
        M.append(m)
    return M

# Account for the distance between two means term
# N - number of points
# M - binary mean variable labels, list of dimenion k x (l * d)
# mi - index of the first mean
# mj - index of the second mean
# P - precision matrix (d x (l * d))
# linear - dictionary of linear factors in QUBO problem
# quadratic - dictionary of quadratic factors in QUBO problem
def btwnMeans(N, M, mi, mj, P, linear, quadratic):
    pp = np.matmul(np.transpose(P), P)  # precision product
    l = len(M[0])
    print("k: " + str(k))

    # account for mi^T P^T P mi term
    for i in range(l):
        for j in range(l):
            if i == j and M[mi][i] in linear:
                linear[M[mi][i]] -= N / k * pp[i][j]
            elif (M[mi][i], M[mi][j]) in quadratic:
                quadratic[(M[mi][i], M[mi][j])] -= N / k * pp[i][j]
            elif i == j:
                linear[M[mi][i]] = N / k * -pp[i][j]
            else:
                quadratic[(M[mi][i], M[mi][j])] = N / k * -pp[i][j]

    # account for 2 mi^T P^T P mj term
    for i in range(l):
        for j in range(l):
            if i == j and mi == mj and M[mi][i] in linear:
                linear[M[mi][i]] += N / k * 2 * pp[i][j]
            elif (M[mi][i], M[mj][j]) in quadratic:
                quadratic[(M[mi][i], M[mj][j])] += N / k * 2 * pp[i][j]
            elif i == j and mi == mj:
                linear[M[mi][i]] = N / k * 2 * pp[i][j]
            else:
                quadratic[(M[mi][i], M[mj][j])] = N / k * 2 * pp[i][j]

    # account for mj^T P^T P mj term
    for i in range(l):
        for j in range(l):
            if i == j and M[mj][i] in linear:
                linear[M[mj][i]] -= N / k * 2 * pp[i][j]
            elif (M[mj][i], M[mj][j]) in quadratic:
                quadratic[(M[mj][i], M[mj][j])] -= N / k * 2 * pp[i][j]
            elif i == j:
                linear[M[mj][i]] = N / k * 2 * -pp[i][j]
            else:
                quadratic[(M[mj][i], M[mj][j])] = N / k * 2 * -pp[i][j]
    
    return linear, quadratic

# Account for the distance from a given mean to a given point (not including constant term)
# M - binary mean variable labels, list of dimenion k x (l * d)
# X - input data
# mi - index of the mean
# xj - index of the data point
# P - precision matrix (d x (l * d))
# linear - dictionary of linear factors in QUBO problem
# quadratic - dictionary of quadratic factors in QUBO problem
def toPoint(M, X, mi, xj, P, linear, quadratic):
    pp = np.matmul(np.transpose(P), P)  # precision product
    l = len(M[0])
    d = np.shape(X)[1]

    # account for mi^T P^T P mi term
    for i in range(l):
        for j in range(l):
            if i == j and M[mi][i] in linear:
                linear[M[mi][i]] += pp[i][j]
            elif (M[mi][i], M[mi][j]) in quadratic:
                quadratic[(M[mi][i], M[mi][j])] += pp[i][j]
            elif i == j:
                linear[M[mi][i]] = pp[i][j]
            else:
                quadratic[(M[mi][i], M[mi][j])] = pp[i][j]

    # account for 2 mi^T P^T P xj term
    for i in range(d):
        for j in range(l):
            if M[mi][j] in linear:
                linear[M[mi][j]] -= 2 * P[i][j] * X[xj][i]
            else:
                linear[M[mi][j]] = -2 * P[i][j] * X[xj][i]

    return linear, quadratic

# Generate QUBO model 
# X - training data
# k - number of clusters
# p - precision vector
def genModel(X, k, p):
    d = np.shape(X)[1]  # dimension of each point/centroid
    l = np.shape(p)[0]  # length of the precision vector
    N = np.shape(X)[0]  # Number of data points
    P = genP(p, d)
    M = genMeanLabels(k, d, l) 

    linear = {}
    quadratic = {}

    # maximize the distance between means
    for i in range(k):
        for j in range(k):
            if i != j:
                linear, quadratic = btwnMeans(N, M, i, j, P, linear, quadratic)

    # minimize the distance between a mean and all surrounding points
    for i in range(k):
        for j in range(N):
            linear, quadratic = toPoint(M, X, i, j, P, linear, quadratic)

    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.Vartype.BINARY)

# Read final assignments of each point
# m - final binary vector (length = l * d * k)
# p - precision vector
# X - input data
# k - number of clusters
def getCentroids(m, p, d, k):
    l = np.shape(p)[0]  # length of the precision vector
    centroids = np.zeros((k, d))
    for i in range(k):
        for j in range(l * d):
            centroids[i][j // l] += int(m["M" + str(i) + "_" + str(j)]) * p[j % l]
    return centroids

def test():
    k = 3
    d = 2
    l = 2
    p = np.transpose(np.array([1, 2]))
    P = genP(p, d)
    M = genMeanLabels(k, d, l)
    mi = 0
    mj = 1
    linear, quadratic = btwnMeans(M, mi, mj, P, linear, quadratic)
    print("Linear: " + str(linear))
    print("\nQuadratic: " + str(quadratic))

if __name__ == "__main__":
    test()