import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

##
# Davis Arthur
# ORNL
# Plotting functions for experiment data
# 7-19-2020
##

def scalability_n():
    classical = np.array([0.02771, 0.16527, 0.61156, 1.99942, 9.27294, 37.10946, 163.44884])
    postprocessing = np.array([0.00018, 0.00036, 0.00092, 0.00204, 0.00191, 0.00365, 0.00757])
    annealing = np.array([0.03481] * len(classical))
    preprocessing = np.array([0.01496, 0.05606, 0.21925, 0.98151, 4.07066, 17.53417, 83.15228])
    errors = [[0.00844, 0.04751, 0.18469, 0.57751, 2.81796, 12.05209, 65.83246], \
        [0.00736, 0.01214, 0.01732, 0.07105, 0.13683, 0.33277, 1.37569]]
    problem_size = ['32', '64', '128', '256', '512', '1024', "2048"]

    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.35
        epsilon = .015
        line_width = 0.5
        opacity = 0.5
        pos_bar_positions = np.arange(len(classical))
        neg_bar_positions = pos_bar_positions + bar_width

        # make bar plots
        classical_bar = plt.bar(pos_bar_positions, classical, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical Algortihm",
                                  yerr = errors[0],
                                  capsize = 2)
        postprocessing_bar = plt.bar(neg_bar_positions, postprocessing, bar_width,
                                  color='white',
                                  label='Postprocessing',
                                  hatch="//",
                                  edgecolor='tab:blue',
                                  ecolor="tab:blue",
                                  linewidth=line_width)
        annealing_bar = plt.bar(neg_bar_positions, annealing, bar_width-epsilon,
                                  bottom=postprocessing,
                                  color="tab:blue",
                                  alpha = opacity,
                                  hatch='//',
                                  edgecolor='tab:blue',
                                  ecolor="tab:blue",
                                  linewidth=line_width,
                                  label='Annealing')
        preprocessing_bar = plt.bar(neg_bar_positions, preprocessing, bar_width-epsilon,
                                   bottom=annealing+postprocessing,
                                   color="tab:blue",
                                   edgecolor="tab:blue",
                                   linewidth=line_width,
                                   label='Preprocessing',
                                   yerr = errors[1],
                                   capsize = 2)
        plt.xticks((neg_bar_positions+pos_bar_positions)/2, problem_size)
        plt.yscale("log")
        plt.ylabel("Time (s)")
        plt.xlabel("Number of points")
        plt.legend(loc = "upper left")
        sns.despine()
        plt.show()

def silhouette_plot():
    labels = ["(16, 2)", "(8, 4)", "(12, 3)", "(15, 3)", "(24, 2)", "(12, 4)", \
        "(21, 3)", "(32, 2)", "(16, 4)"]
    classical = [0.78479498, 0.66593922, 0.72548824, 0.73731669, 0.8119265, \
        0.69552446, 0.74917315, 0.81160783, 0.69397682]
    quantum = [0.74705008, 0.71771498, 0.75313274, 0.7208896, 0.77899294, \
        0.5813618, 0.69856036, 0.75908969, 0.53162158]
    errors = [[0.12865615, 0.1929148, 0.17101924, 0.10874116, 0.10101152, \
        0.10352797, 0.11773225, 0.11389882, 0.15616958], \
        [0.17935095, 0.09859737, 0.09625055, 0.12834223, 0.14642393, \
        0.21007092, 0.19238295, 0.17047154, 0.21443379]]

    x = 1.5 * np.arange(len(labels))  # the label locations
    width = .3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2.0, classical, width, label = "Classical Algorithm", \
        yerr = errors[0], capsize = 2)
    rects2 = ax.bar(x + width / 2.0, quantum, width, label = "Quantum Algorithm", \
        yerr = errors[1], capsize = 2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Average Silhouette Distance")
    ax.set_xlabel("(N, k)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower left")
    fig.tight_layout()
    plt.show()

def iris_plot():
    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)
        labels = ["(9, 3)", "(14, 2)", "(15, 3)", "(24, 2)", "(21, 3)", "(32, 2)"]
        classical = [0.7504, 0.86597938, 0.72388889, 0.90150143, 0.76998492, 0.9393346]
        quantum = [0.74902373, 0.7662167, 0.69122507, 0.71497528, 0.67558283, 0.6749335]
        errors = [[0.1728, 0.0, 0.16907724, 0.12851441, 0.16612592, 0.0], \
            [0.17235436, 0.12465246, 0.16098572, 0.14788062, 0.18368838, 0.1222211]]

        x = 1.5 * np.arange(len(labels))  # the label locations
        width = .3  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2.0 * 1.08, classical, width, label = "Classical Algorithm", edgecolor='tab:blue')
        rects2 = ax.bar(x + width / 2.0 * 1.08, quantum, width, label = "Quantum Algorithm", edgecolor='tab:orange')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Adjusted Rand Index")
        ax.set_xlabel("(N, k)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc = "upper left")
        fig.tight_layout()
        sns.despine()
        plt.show()

def time_plot():
    labels = ["64", "128", "256", "512", "1024", "2056"]
    classical = [0.1168, 0.5169, 2.1566, 8.5344, 36.1996, 182.4552]
    quantum = [0.7323, 0.9017, 1.6966, 5.1333, 19.0854, 100.7574]
    errors = [[0.0334, 0.1839, 0.6843, 2.3419, 11.4519, 64.9139], \
    [0.1754, 0.1736, 0.2139, 0.2606, 0.4608, 19.5783]]

    x = 1.5 * np.arange(len(labels))  # the label locations
    width = 1.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, classical, width, label = "Classical Algorithm", \
        yerr = errors[0], capsize = 2)
    rects2 = ax.bar(x + width, quantum, width, label = "Quantum Annealing", \
        yerr = errors[1], capsize = 2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Number of Datapoints (N)")
    plt.yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fig.tight_layout()
    plt.show()

def demo_classical():
    N = 120
    k = 3
    d = 2
    X = genData(N, k, d)[0]
    plt.rcParams.update({'font.size': 14})
    for i in range(N):
        plt.scatter(X[i][0], X[i][1], c = "k")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

    assignments = balanced.balanced_kmeans(X, k)[1]
    colors = ["tab:orange", "tab:blue", "tab:green"]
    for i in range(N):
        plt.scatter(X[i][0], X[i][1], c = colors[assignments[i]])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    patches = []
    patches.append(mpatches.Patch(color = colors[0], label = "Cluster 1"))
    patches.append(mpatches.Patch(color = colors[1], label = "Cluster 2"))
    patches.append(mpatches.Patch(color = colors[2], label = "Cluster 3"))
    plt.legend(handles = patches, loc = "lower left")
    plt.show()

def quantum_demo():
    N = 21
    k = 3
    d = 2
    X = np.array([[6.69949807, -8.47161991], [ -3.58502492, -7.00267002], \
    [6.9883606, -8.11954925], [-3.10593036, -9.43520019], [-3.7161674, -1.49098871], \
    [-2.09338464, -8.6062541], [-0.85485456, -10.4618202], [-2.94110897, -3.49214472], \
    [-3.75253146, -2.33793164], [-1.20518409, -3.32417214], [-4.94279692, -7.53898352], \
    [-3.13245473, -3.17353815], [-4.24776418, -5.96847917], [2.91528502, -6.50722037], \
    [-1.33137606, -2.74582], [-3.6195642, -2.76961647], [5.67383599, -7.60517892], \
    [5.71101514, -8.45136874], [  6.00095131, -7.04700216], [-3.61318392, -7.76251629], \
    [5.52186512, -8.26647578]])
    plt.rcParams.update({'font.size': 14})
    for i in range(N):
        plt.scatter(X[i][0], X[i][1], c = "k")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

    assignments = balanced.balanced_kmeans(X, k)[1]
    colors = ["tab:orange", "tab:blue", "tab:green"]
    for i in range(N):
        plt.scatter(X[i][0], X[i][1], c = colors[assignments[i]])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    patches = []
    patches.append(mpatches.Patch(color = colors[0], label = "Cluster 1"))
    patches.append(mpatches.Patch(color = colors[1], label = "Cluster 2"))
    patches.append(mpatches.Patch(color = colors[2], label = "Cluster 3"))
    plt.legend(handles = patches, loc = "upper right")
    plt.show()
