import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

##
# Davis Arthur
# ORNL
# Plotting functions for experiment data
# 7-19-2020
##

def scalability_n():
    classical = np.array([0.021762929, 0.025610843, 0.033377123, 0.041359906, 0.052094159, 0.068444805, 0.100637832])
    balanced = np.array([0.002784061, 0.007340279, 0.031504641, 0.163723245, 1.557713284, 10.8927644, 95.48761162])
    postprocessing = np.array([0.00018, 0.00036, 0.00092, 0.00204, 0.00191, 0.00365, 0.00757])
    annealing = np.array([0.03481] * len(classical))
    qubo_formulation = np.array([0.000805612, 0.007043934, 0.01922039, 0.115365491, 0.462354026, 1.840909271, 7.690179176])
    embedding = np.array([0.125221772, 0.497308581, 1.983284184, 7.922443327, 31.66959336, 126.6392204, 506.4797824])
    errors = [[0.00173509, 0.002211505, 0.003549236, 0.005998092, 0.008509194, 0.013434674, 0.023120421], \
        [0.000771075, 0.002484552, 0.014267903, 0.060656692, 1.050099485, 5.540460328, 58.01033059], \
        [0.000336993, 0.000833211, 0.001798659, 0.002392254, 0.009510396, 0.020083551, 0.058115987]]
    problem_size = ['64', '128', '256', '512', '1024', "2048", "4096"]

    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        classical_positions = np.arange(len(classical)) * 2
        balanced_positions = classical_positions + bar_width * 1.04
        quantum_positions = classical_positions + bar_width * 2 * 1.04

        # make bar plots
        classical_bar = plt.bar(classical_positions, classical, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "k",
                                  yerr = errors[0],
                                  capsize = 2)
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="k",
                                  label = "Classical balanced k-means",
                                  yerr = errors[1],
                                  capsize = 2)
        postprocessing_bar = plt.bar(quantum_positions, postprocessing, bar_width,
                                  color='white',
                                  label='Postprocessing',
                                  edgecolor='k',
                                  ecolor="tab:blue",
                                  linewidth=line_width)
        annealing_bar = plt.bar(quantum_positions, annealing, bar_width,
                                  bottom=postprocessing,
                                  color="tab:blue",
                                  alpha = opacity,
                                  hatch='//',
                                  edgecolor='k',
                                  ecolor="tab:blue",
                                  linewidth=line_width,
                                  label='Annealing')
        qubo_formulation_bar = plt.bar(quantum_positions, qubo_formulation, bar_width,
                                   bottom=annealing+postprocessing,
                                   color="tab:blue",
                                   edgecolor="k",
                                   linewidth=line_width,
                                   label='QUBO formulation')
        embedding_bar = plt.bar(quantum_positions, embedding, bar_width,
                                  bottom=qubo_formulation+annealing+postprocessing,
                                  color="tab:blue",
                                  alpha = opacity * 2.0,
                                  hatch='//',
                                  edgecolor='k',
                                  ecolor="k",
                                  linewidth=line_width,
                                  label='Embedding',
                                  yerr = errors[2],
                                  capsize = 2)
        plt.xticks((quantum_positions+classical_positions)/2, problem_size)
        plt.yscale("log")
        plt.ylabel("Time (s)")
        plt.xlabel("Number of points")
        plt.legend(loc = "upper left")
        sns.despine()
        plt.show()

def scalability_k():
    classical = np.array([0.027070456, 0.042738562, 0.058373733, 0.087332768, 0.13494369, 0.234100118])
    balanced = np.array([0.027599249, 0.041715598, 0.039936337, 0.038980017, 0.027072248, 0.020074463])
    postprocessing = np.array([0.001088924, 0.00096468, 0.00090342, 0.000981359, 0.000866399, 0.000991993])
    annealing = np.array([0.03481] * len(classical))
    qubo_formulation = np.array([0.007992764, 0.019775014, 0.12734726, 0.512867808, 1.959843183, 7.651099477])
    embedding = np.array([0.497308581, 1.983284184, 7.922443327, 31.66959336, 126.6392204, 506.4797824])
    errors = [[0.004198251, 0.005792501, 0.005240937, 0.008861572, 0.012026878, 0.009035137], \
        [0.010074777, 0.016354764, 0.011292965, 0.00940675, 0.0052254, 0.002333708], \
        [0.001081033, 0.002058181, 0.005319191, 0.027136404, 0.030796908, 0.069548552]]
    problem_size = ["2", "4", "8", "16", "32", '64']

    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        classical_positions = np.arange(len(classical)) * 2
        balanced_positions = classical_positions + bar_width * 1.04
        quantum_positions = classical_positions + bar_width * 2 * 1.04

        # make bar plots
        classical_bar = plt.bar(classical_positions, classical, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "k",
                                  yerr = errors[0],
                                  capsize = 2)
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="k",
                                  label = "Classical balanced k-means",
                                  yerr = errors[1],
                                  capsize = 2)
        postprocessing_bar = plt.bar(quantum_positions, postprocessing, bar_width,
                                  color='white',
                                  label='Postprocessing',
                                  hatch="//",
                                  edgecolor='k',
                                  ecolor="tab:blue",
                                  linewidth=line_width)
        annealing_bar = plt.bar(quantum_positions, annealing, bar_width,
                                  bottom=postprocessing,
                                  color="tab:blue",
                                  alpha = opacity,
                                  hatch='//',
                                  edgecolor='k',
                                  ecolor="tab:blue",
                                  linewidth=line_width,
                                  label='Annealing')
        qubo_formulation_bar = plt.bar(quantum_positions, qubo_formulation, bar_width,
                                   bottom=annealing+postprocessing,
                                   color="tab:blue",
                                   edgecolor="k",
                                   linewidth=line_width,
                                   label='QUBO formulation')
        embedding_bar = plt.bar(quantum_positions, embedding, bar_width,
                                  bottom=qubo_formulation+annealing+postprocessing,
                                  color="tab:blue",
                                  alpha = opacity * 2.0,
                                  hatch='//',
                                  edgecolor='k',
                                  ecolor="k",
                                  linewidth=line_width,
                                  label='Embedding',
                                  yerr = errors[2],
                                  capsize = 2)
        plt.xticks((quantum_positions+classical_positions)/2, problem_size)
        plt.yscale("log")
        plt.ylabel("Time (s)")
        plt.xlabel("Number of clusters (k)")
        plt.legend(bbox_to_anchor=(0.48, 0.68))
        sns.despine()
        plt.show()

def scalability_d():
    classical = np.array([0.050836406, 0.068124766, 0.08031024, 0.418972721, 0.541129889, 0.659788241, 1.036860147, 1.247389317])
    balanced = np.array([1.506752596, 1.758921628, 2.259125872, 2.067227983, 2.159853106, 1.898321447, 1.676846294, 1.406032372])
    postprocessing = np.array([0.00368886, 0.003750648, 0.003532062, 0.003698163, 0.003639903, 0.003679199, 0.003725591, 0.003911486])
    annealing = np.array([0.03481] * len(classical))
    qubo_formulation = np.array([0.474166341, 0.477093368, 0.473733292, 0.476038346, 0.489536886, 0.503299804, 0.528266611, 0.57594285])
    embedding = np.array([31.66959336] * len(classical))
    errors = [[0.008908656, 0.013745631, 0.010549941, 0.106498582, 0.117121529, 0.104829964, 0.157666514, 0.172574078], \
        [0.689887604, 0.654587318, 1.043548052, 0.647316611, 0.615670114, 0.488792789, 0.455110754, 0.218427744], \
        [0.018545238, 0.018880272, 0.010093438, 0.010171969, 0.017934214, 0.017834971, 0.022132621, 0.020678549]]
    problem_size = ["2", "4", "8", "16", "32", '64', '128', '256']

    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        classical_positions = np.arange(len(classical)) * 2
        balanced_positions = classical_positions + bar_width * 1.04
        quantum_positions = classical_positions + bar_width * 2 * 1.04

        # make bar plots
        classical_bar = plt.bar(classical_positions, classical, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "k",
                                  yerr = errors[0],
                                  capsize = 2)
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="k",
                                  label = "Classical balanced k-means",
                                  yerr = errors[1],
                                  capsize = 2)
        postprocessing_bar = plt.bar(quantum_positions, postprocessing, bar_width,
                                  color='white',
                                  label='Postprocessing',
                                  hatch="//",
                                  edgecolor='k',
                                  ecolor="tab:blue",
                                  linewidth=line_width)
        annealing_bar = plt.bar(quantum_positions, annealing, bar_width,
                                  bottom=postprocessing,
                                  color="tab:blue",
                                  alpha = opacity,
                                  hatch='//',
                                  edgecolor='k',
                                  ecolor="tab:blue",
                                  linewidth=line_width,
                                  label='Annealing')
        qubo_formulation_bar = plt.bar(quantum_positions, qubo_formulation, bar_width,
                                   bottom=annealing+postprocessing,
                                   color="tab:blue",
                                   edgecolor="k",
                                   linewidth=line_width,
                                   label='QUBO formulation')
        embedding_bar = plt.bar(quantum_positions, embedding, bar_width,
                                  bottom=qubo_formulation+annealing+postprocessing,
                                  color="tab:blue",
                                  alpha = opacity * 2.0,
                                  hatch='//',
                                  edgecolor='k',
                                  ecolor="k",
                                  linewidth=line_width,
                                  label='Embedding',
                                  yerr = errors[2],
                                  capsize = 2)
        plt.xticks((quantum_positions+classical_positions)/2, problem_size)
        plt.yscale("log")
        plt.ylabel("Time (s)")
        plt.ylim(top=1000)
        plt.xlabel("Number of features (d)")
        plt.legend(bbox_to_anchor=(0.09, 0.75))
        sns.despine()
        plt.show()

def rand_synth_plot():
    sklearn = np.array([0.580094883, 0.652637425, 0.597850414, 0.492597192, 0.516493305, 0.55340865, 0.418795854, 0.431164763, 0.434192296])
    balanced = np.array([0.664930721, 0.658507942, 0.645040003, 0.536853906, 0.558718152, 0.567830142, 0.47081761, 0.485086364, 0.444471154])
    quantum = np.array([0.449401984, 0.456969746, 0.326448614, 0.529441471, 0.593871239, 0.45675419, 0.58341195, 0.479643148, 0.347148747])
    errors = [[0.315202947, 0.26, 0.268112452, 0.240737079, 0.266576692, 0.24304026, 0.230263959, 0.256670794, 0.169500821], \
        [0.370334405, 0.4463540, 0.3, 0.315956619, 0.294181131, 0.342250967, 0.260361111, 0.296400221, 0.266253438], \
        [0.3, 0.442313518, 0.276164397, 0.270820128, 0.266502967, 0.298324165, 0.211407229, 0.200233215, 0.201451633]]
    labels = ["(16, 2)", "(24, 2)", "(32, 2)", "(12, 3)", "(15, 3)", "(21, 3)", "(8, 4)", "(12, 4)", "(16, 4)"]

    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        sklearn_positions = np.arange(len(sklearn)) * 2
        balanced_positions = sklearn_positions + bar_width * 1.04
        quantum_positions = sklearn_positions + bar_width * 2 * 1.04

        # make bar plots
        sklearn_bar = plt.bar(sklearn_positions, sklearn, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "k")
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="k",
                                  label = "Classical balanced k-means")
        quantum_bar = plt.bar(quantum_positions, quantum, bar_width,
                                  color='tab:blue',
                                  label='Quantum balanced k-means',
                                  edgecolor='k',
                                  ecolor="k",
                                  linewidth=line_width)
        plt.xticks((quantum_positions+sklearn_positions)/2, labels)
        plt.ylabel("Adjusted Rand Index")
        plt.xlabel("(Number of points, Number of clusters)")
        plt.legend(bbox_to_anchor=(0.6, 0.90))
        sns.despine()
        plt.show()

def silhouette_synth_plot():
    sklearn = np.array([0.541433302, 0.449842675, 0.49954606, 0.505311666, 0.548444319, 0.489588789, 0.504470407, 0.542066952, 0.481403321])
    balanced = np.array([0.4671494, 0.241690426, 0.339342297, 0.3789799, 0.472987593, 0.3, 0.40954325, 0.484606704, 0.321982645])
    quantum = np.array([0.405615479, 0.349731418, 0.3852053, 0.407101012, 0.4, 0.337223155, 0.345728337, 0.308908142, 0.229299506])
    errors = [[0.096417547, 0.11, 0.083327477, 0.079623158, 0.087937082, 0.094698241, 0.065330361, 0.069663618, 0.061758381], \
        [0.119455045, 0.2, 0.129184931, 0.131238373, 0.106709549, 0.14493716, 0.102858038, 0.094538557, 0.113685937], \
        [0.135662267, 0.166353284, 0.13740935, 0.124135567, 0.137804989, 0.152598362, 0.155208096, 0.141087555, 0.164753283]]
    labels = ["(16, 2)", "(8, 4)", "(12, 3)", "(15, 3)", "(24, 2)", "(12, 4)", "(21, 3)", "(32, 2)", "(16, 4)"]
    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        sklearn_positions = np.arange(len(sklearn)) * 2
        balanced_positions = sklearn_positions + bar_width * 1.04
        quantum_positions = sklearn_positions + bar_width * 2 * 1.04

        # make bar plots
        sklearn_bar = plt.bar(sklearn_positions, sklearn, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "tab:orange",
                                  yerr = errors[0],
                                  capsize = 2)
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="tab:green",
                                  label = "Classical balanced k-means",
                                  yerr = errors[1],
                                  capsize = 2)
        quantum_bar = plt.bar(quantum_positions, quantum, bar_width,
                                  color='tab:blue',
                                  label='Quantum balanced k-means',
                                  edgecolor='tab:blue',
                                  ecolor="k",
                                  linewidth=line_width,
                                  yerr = errors[2],
                                  capsize = 2)
        plt.xticks((quantum_positions+sklearn_positions)/2, labels)
        plt.ylabel("Silhouette")
        plt.xlabel("(Number of points, Number of clusters)")
        plt.legend(loc = "lower right")
        sns.despine()
        plt.show()

def rand_iris_plot():
    sklearn = np.array([0.774193548, 0.881889764, 0.919860627, 0.939334638, 0.649117081, 0.691091292, 0.725265438, 0.701017159, 0.690725755])
    balanced = np.array([0.77, 0.9, 0.9, 0.939334638, 7.57E-01, 7.50E-01, 0.762777778, 0.75337799, 7.72E-01])
    quantum = np.array([0.75, 0.721381067, 0.727640916, 0.657951871, 7.42E-01, 7.33E-01, 7.24E-01, 6.96E-01, 6.40E-01])
    errors = [[0, 0, 0, 0, 0.170107997, 0.139872304, 0.129912586, 0.145516456, 0.117441293], \
        [0, 0, 0, 0, 0.178856814, 0.183764537, 0.179700754, 0.178424848, 0.146908636], \
        [0.085633239, 0.133999257, 0.121523807, 0.118, 0.186733529, 0.185251731, 0.170182799, 0.161791347, 0.139134728]]
    labels = ["(8, 2)", "(16, 2)", "(24, 2)", "(32, 2)", "(9, 3)", "(12, 3)", "(15, 3)", "(18, 3)", "(21, 3)"]
    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        sklearn_positions = np.arange(len(sklearn)) * 2
        balanced_positions = sklearn_positions + bar_width * 1.04
        quantum_positions = sklearn_positions + bar_width * 2 * 1.04

        # make bar plots
        sklearn_bar = plt.bar(sklearn_positions, sklearn, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "k")
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="k",
                                  label = "Classical balanced k-means")
        quantum_bar = plt.bar(quantum_positions, quantum, bar_width,
                                  color='tab:blue',
                                  label='Quantum balanced k-means',
                                  edgecolor='k',
                                  ecolor="k",
                                  linewidth=line_width)
        plt.xticks((quantum_positions+sklearn_positions)/2, labels)
        plt.ylabel("Adjusted Rand Index")
        plt.xlabel("(Number of points, Number of clusters)")
        plt.legend(loc = "lower right")
        sns.despine()
        plt.show()

def silhouette_iris_plot2():
    sklearn = np.array([0.74, 0.734320229, 0.745161564, 0.7])
    balanced = np.array([0.741886373, 0.734320229, 0.745161564, 0.746486183])
    quantum = np.array([0.732761352, 0.658426678, 0.655325653, 0.60889868])
    errors = [[0.059324419, 0.038743111, 0.028724049, 2.25E-02], \
        [0.059324419, 0.038743111, 0.028724049, 0.022547611], \
        [0.081125425, 0.083970171, 0.06970543, 0.068171983]]
    labels = ["(8, 2)", "(16, 2)", "(24, 2)", "(32, 2)"]
    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        sklearn_positions = np.arange(len(sklearn)) * 2
        balanced_positions = sklearn_positions + bar_width * 1.04
        quantum_positions = sklearn_positions + bar_width * 2 * 1.04

        # make bar plots
        sklearn_bar = plt.bar(sklearn_positions, sklearn, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "tab:orange",
                                  yerr = errors[0],
                                  capsize = 2)
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="tab:green",
                                  label = "Classical balanced k-means",
                                  yerr = errors[1],
                                  capsize = 2)
        quantum_bar = plt.bar(quantum_positions, quantum, bar_width,
                                  color='tab:blue',
                                  label='Quantum balanced k-means',
                                  edgecolor='tab:blue',
                                  ecolor="k",
                                  linewidth=line_width,
                                  yerr = errors[2],
                                  capsize = 2)
        plt.xticks((quantum_positions+sklearn_positions)/2, labels)
        plt.ylabel("Silhouette")
        plt.xlabel("(Number of points, Number of clusters)")
        plt.legend(loc = "lower right")
        sns.despine()
        plt.show()

def rand_iris_plot3():
    sklearn = np.array([0.649117081, 0.691091292, 0.725265438, 0.701017159, 0.690725755])
    balanced = np.array([7.57E-01, 7.50E-01, 0.762777778, 0.75337799, 7.72E-01])
    quantum = np.array([7.42E-01, 7.33E-01, 7.24E-01, 6.96E-01, 6.40E-01])
    errors = [[0.170107997, 0.139872304, 0.129912586, 0.145516456, 0.117441293], \
        [0.178856814, 0.183764537, 0.179700754, 0.178424848, 0.146908636], \
        [0.186733529, 0.185251731, 0.170182799, 0.161791347, 0.139134728]]
    labels = ["(9, 3)", "(12, 3)", "(15, 3)", "(18, 3)", "(21, 3)"]
    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        sklearn_positions = np.arange(len(sklearn)) * 2
        balanced_positions = sklearn_positions + bar_width * 1.04
        quantum_positions = sklearn_positions + bar_width * 2 * 1.04

        # make bar plots
        sklearn_bar = plt.bar(sklearn_positions, sklearn, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "tab:orange",
                                  yerr = errors[0],
                                  capsize = 2)
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="tab:green",
                                  label = "Classical balanced k-means",
                                  yerr = errors[1],
                                  capsize = 2)
        quantum_bar = plt.bar(quantum_positions, quantum, bar_width,
                                  color='tab:blue',
                                  label='Quantum balanced k-means',
                                  edgecolor='tab:blue',
                                  ecolor="k",
                                  linewidth=line_width,
                                  yerr = errors[2],
                                  capsize = 2)
        plt.xticks((quantum_positions+sklearn_positions)/2, labels)
        plt.ylabel("Adjusted Rand Index")
        plt.xlabel("(Number of points, Number of clusters)")
        plt.legend(loc = "lower right")
        sns.despine()
        plt.show()

def silhouette_iris_plot3():
    sklearn = np.array([5.87E-01, 6.01E-01, 5.72E-01, 5.72E-01, 5.70E-01])
    balanced = np.array([5.20E-01, 5.28E-01, 5.13E-01, 5.13E-01, 5.10E-01])
    quantum = np.array([0.525794253, 0.537768665, 5.16E-01, 4.97E-01, 0.449877595])
    errors = [[0.069402846, 0.062313865, 0.05465138, 0.05936624, 0.042606769], \
        [0.108282059, 0.071424374, 0.071128219, 0.067908071, 0.041732992], \
        [0.109951073, 0.072290149, 0.079936002, 0.101453908, 0.093712221]]
    labels = ["(9, 3)", "(12, 3)", "(15, 3)", "(18, 3)", "(21, 3)"]
    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context(font_scale = 2.5)

        # plot details
        bar_width = 0.36
        epsilon = .015
        line_width = 0.5
        opacity = 0.33
        sklearn_positions = np.arange(len(sklearn)) * 2
        balanced_positions = sklearn_positions + bar_width * 1.04
        quantum_positions = sklearn_positions + bar_width * 2 * 1.04

        # make bar plots
        sklearn_bar = plt.bar(sklearn_positions, sklearn, bar_width,
                                  color='tab:orange',
                                  linewidth=line_width,
                                  label = "Classical k-means",
                                  edgecolor = "tab:orange",
                                  yerr = errors[0],
                                  capsize = 2)
        balanced_bar = plt.bar(balanced_positions, balanced, bar_width,
                                  color='tab:green',
                                  linewidth=line_width,
                                  edgecolor="tab:green",
                                  label = "Classical balanced k-means",
                                  yerr = errors[1],
                                  capsize = 2)
        quantum_bar = plt.bar(quantum_positions, quantum, bar_width,
                                  color='tab:blue',
                                  label='Quantum balanced k-means',
                                  edgecolor='tab:blue',
                                  ecolor="k",
                                  linewidth=line_width,
                                  yerr = errors[2],
                                  capsize = 2)
        plt.xticks((quantum_positions+sklearn_positions)/2, labels)
        plt.ylabel("Silhouette")
        plt.xlabel("(Number of points, Number of clusters)")
        plt.legend(loc = "lower right")
        sns.despine()
        plt.show()

def embedding_time():
    # plot actual data
    num_var = []
    for i in range(4, 17):
        num_var.append(i * 4)
    embedding_time = np.array([0.000986514, 0.001197014, 0.001573615, 0.002077231, \
        0.002473483, 0.00303431, 0.003646517, 0.004218731, 0.00487071, 0.005742383, \
        0.006704602, 0.00743187, 0.008415136]) * 1000
    for i in range(len(num_var)):
        plt.scatter(num_var[i], embedding_time[i], c = "k")

    # plot line of best fit
    num_points = 500
    first = 10.0
    last = 70.0
    deltaX = (last - first) / num_points
    X = []
    Y = []
    for i in range(num_points):
        X.append(first + i * deltaX)
        Y.append((1.8865E-06 * X[i] ** 2.0 + 4.6321E-06 * X[i] + 4.0229E-04) * 1000)
    plt.plot(X, Y, label = "$0.001887 (Nk)^2 + 0.004632Nk + 0.4022$")
    plt.legend(loc = "upper left")
    plt.xlabel("Number of binary variables ($Nk$)")
    plt.ylabel("Time (ms)")
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
    N = 15
    k = 3
    d = 2
    X = np.array([[-0.90072912, -0.32418852], [-1.51689349, -1.09179728], \
        [-1.60265074,  0.70956788], [-0.94574821,  0.93750845], \
        [-0.91106794, -1.08365852], [ 0.42258706, -2.13386422], \
        [ 0.64354346, -1.41450107], [-2.40999261,  0.31022965], \
        [-1.49334317, -1.55073057], [-0.16246154,  1.47937435], \
        [-0.9662797,   1.00253344], [-0.64615159, -0.59283228], \
        [ 1.0405294,  -0.82615799], [ 0.39381428, -1.42601059], [0.18708717, -1.52117236]])
    plt.rcParams.update({'font.size': 14})
    colors = ["tab:orange", "tab:blue", "tab:green"]
    target = [0, 0, 2, 2, 0, 1, 1, 2, 0, 2, 2, 0, 1, 1, 1]
    for i in range(N):
        plt.scatter(X[i][0], X[i][1], c = colors[target[i]])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    patches = []
    patches.append(mpatches.Patch(color = colors[0], label = "Class 1"))
    patches.append(mpatches.Patch(color = colors[1], label = "Class 2"))
    patches.append(mpatches.Patch(color = colors[2], label = "Class 3"))
    plt.legend(handles = patches, loc = "upper right")
    plt.show()

    colors = ["tab:orange", "tab:blue", "tab:green"]
    assignments = [0, 0, 2, 2, 0, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1]
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

rand_synth_plot()
