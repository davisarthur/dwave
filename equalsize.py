import numpy as np
import random
import dimod

##
# Davis Arthur
# ORNL
# Equal size k-means clustering problem as QUBO problem
# 6-17-2020
##

# Generate F
# N - number of points
# k - number of clusters
def genF(N, k):
    return np.ones((N, N)) - 2 * N / k * np.identity(N)

# Generate G
# N - number of points
# k - number of clusters
def genG(N, k):
    return np.ones((k, k)) - 2 * np.identity(k)

# Generate D
# X - training data
def genD(X):
    N = np.shape(X)[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i][j] = np.linalg.norm(X[i] - X[j])
    return D
 
# Generate qubo model
# X - training data
# k - number of clusters
# p1 - penalty multiplier for column constraints
# p2 - penalty multiplier for row constraints
def genModel(X, k, p1 = 10.0, p2 = 10.0):
    N = np.shape(X)[0]      # number of points
    D = genD(X)             # distance matrix
    F = p1 * genF(N, k)     # column penalty matrix
    G = p2 * genG(N, k)     # row penalty matrix

    # create array of binary variable labels
    W = []
    for i in range(N):
        row = []
        for j in range(k):
            row.append("w" + str(i) + str(j))
        W.append(row)

    linear = {}
    quadratic = {}

    # account for D term
    for l in range(k):
        for j in range(N):
            for i in range(N):
                if W[i][l] in linear:
                    linear[W[i][l]] = linear[W[i][l]] + D[i][j]
                elif (W[i][l], W[j][l]) in quadratic:
                    quadratic[(W[i][l], W[j][l])] = quadratic[(W[i][l], W[j][l])] + D[i][j]
                elif i == j:
                    linear[W[i][l]] = D[i][j]
                else:
                    quadratic[(W[i][l], W[j][l])] = D[i][j]

    # account for F term
    for l in range(k):
        for j in range(N):
            for i in range(N):
                if W[i][l] in linear:
                    linear[W[i][l]] = linear[W[i][l]] + F[i][j]
                elif (W[i][l], W[j][l]) in quadratic:
                    quadratic[(W[i][l], W[j][l])] = quadratic[(W[i][l], W[j][l])] + F[i][j]
                elif i == j:
                    linear[W[i][l]] = F[i][j]
                else:
                    quadratic[(W[i][l], W[j][l])] = F[i][j]

    # account for G term
    for l in range(N):
        for j in range(k):
            for i in range(k):
                if W[l][i] in linear:
                    linear[W[l][i]] = linear[W[l][i]] + G[i][j]
                elif (W[l][i], W[l][j]) in quadratic:
                    quadratic[(W[l][i], W[l][j])] = quadratic[(W[l][i], W[l][j])] + G[i][j]
                elif i == j:
                    linear[W[l][i]] = G[i][j]
                else:
                    quadratic[(W[l][i], W[l][j])] = G[i][j]

    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.Vartype.BINARY)

# Generate random training data matrix
# N - number of points
# d - dimension of each point
def genData(N, d):
    return np.random.random((N, d))

# Test case for k = 2 clustering
def test():
    X = np.array([[1, 2], [1, 3], [5, 1], [6, 2], [3, 2], [4, 2]])
    model = genModel(X, 2)
    print(model)
    sampleset = dimod.ExactSolver().sample_qubo(model)
    print(sampleset.first.sample)

# 2d dimensional example
def example2d():
    # prompt user
    N = int(input("Number of points: "))
    k = int(input("Number of clusters: "))

    # calculate solution
    d = 2
    X = genData(N, d)
    model = genModel(X, k)
    qubo, offset = model.to_qubo()
    solution = solve_qubo(qubo)

    # plot the solution
    meanColors = ["c", "y", "m", "g"]
    for i in range(N):
        for j in range(k):
            if solution["w[" + str(i) + "][" + str(j) + "]"] != 0:
                plt.scatter(X[i][0], X[i][1], c = meanColors[j])

    plt.title("Equal Size $k$-means clustering ($k = " + str(k) + "$)")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

if __name__ == "__main__":
    test()
