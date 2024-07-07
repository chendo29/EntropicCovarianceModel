"""

This module contains an example on the usage of Entropic Covariance Model.

"""
import numpy as np
import time
from matplotlib import pyplot as plt

from EntropicCovModel import EntropicCovModel, LinkFunctionFactory

if __name__ == "__main__":
    """
    Identical to the second toy test in terms of set up. We then compare the 
    performance of models using the SMSI link function and the log link funciton.
    The difference is that optimization is performed using Stochastic Gradient
    Descent.
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[-5 - x_i, -3 + x_i],
              [-3 + x_i, -3 + x_i]]

    Target alpha is [-5, -1, -3, 1, -3, 1]
    """

    np.random.seed(1226789)

    num_samples = 50000
    X = np.random.normal(5, 1, num_samples)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "SGD"
    initial_guess = 10 * np.random.rand(6)
    learning_rate = 0.01
    num_iterations = 50
    #batch_size = 10



    batch_sizes = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    train_time = []

    for batch_size in batch_sizes:
        print('batch size: ' + str(batch_size))
        start_time = time.time()
        optimization_config_model1 = {"initial_guess": initial_guess,
                                      "learning_rate": learning_rate,
                                      "n_iter": num_iterations,
                                      "tolerance": 1e-05,
                                      "batch_size": batch_size,
                                      "num_samples": num_samples}
        model = EntropicCovModel("example_feature_map_2",
                                 "SMSI", X, Y,
                                 optimizer_type, optimization_config_model1)

        est_alpha = model.fit()
        end_time = time.time()
        elapsed_time = end_time - start_time
        train_time.append(elapsed_time)

    labels = [str(batch_size) for batch_size in batch_sizes]
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.5
    plt.bar(labels, train_time, color='skyblue', width=bar_width)

    # Add title and labels
    plt.title('Training Time for 50 iterations vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Training Time')

    plt.savefig('train_time_vs_batch_size_50k_samples.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.close()
