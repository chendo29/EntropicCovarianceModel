"""

This module contains demos on the usage of Entropic Covariance Model.

"""
import numpy as np

import matplotlib.pyplot as plt

from EntropicCovModel import EntropicCovModel, LinkFunctionFactory

if __name__ == "__main__":
    """
    First toy test is to simulate data from 4-d multivariate normal distribution
    The simulation generates a sequence of 4-tuple (z_1,z_2,z_3,z_4). Let
        i) x_i = (z_1,z_2)
        ii) y_i = (z_3,z_4)
    Then, we know y_i|x_i=a follows a MVN(m,C), where 
        i) m = cov(y_i,x_i)[cov(x_i,x_i)]^{-1}a
        ii) C = cov(y_i,y_i) - cov(y_i,x_i)cov(x_i,x_i)^{-1}cov(x_i,y_i)
    """

    # set seed to preserve the simulation results
    np.random.seed(1226789)

    # Define the mean vector (4-dimensional)
    mean = [0, 0, 0, 0]

    # Define the covariance matrix (4x4)
    Sigma = [[1, 0.5, 0.3, 0.1],
             [0.5, 1, 0.2, 0.4],
             [0.3, 0.2, 1, 0.3],
             [0.1, 0.4, 0.3, 1]]

    # Block of cov matrix
    Sigma11 = [[1, 0.5],
               [0.5, 1]]
    Sigma12 = [[0.3, 0.1],
               [0.2, 0.4]]
    Sigma21 = [[0.3, 0.2],
               [0.1, 0.4]]
    Sigma22 = [[1, 0.3],
               [0.3, 1]]

    # Target Cov matrix, this is the target matrix that we want to approximate
    C = Sigma22 - Sigma21 @ np.linalg.inv(Sigma11) @ Sigma12

    # Generate 1000 samples from the multivariate Gaussian distribution
    samples = np.random.multivariate_normal(mean, Sigma, 1000)
    X = samples[:, :2]
    Y = samples[:, -2:]

    # Center Y|X
    mean_Y_cond = np.array([Sigma21 @ np.linalg.inv(Sigma11) @ x for x in X])
    Y = Y - mean_Y_cond

    print("Generated Samples:")
    print(samples)
    print("Generated X:")
    print(X)
    print("Generated Y:")
    print(Y)

    # Construct the Entropic Covariance Model
    initial_guess = 100 * np.random.rand(9)
    learning_rate = 0.0001
    num_iterations = 500
    optimizer_type = "GradientDescent"
    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate": learning_rate,
                           "num_of_iteration": num_iterations,
                           "tolerance": 1e-05}
    model = EntropicCovModel("example_feature_map_1", "SMSI",
                             X, Y, optimizer_type, optimization_config)

    est_alpha = model.fit()
    print(est_alpha)
    # Print Estimate for one sample for readability
    print("1st Target C:")
    print(C)
    print("1st Estimated C:")
    print(model.get_estimate(est_alpha[-1])[0])

if __name__ == "__main__":
    """
    Second Toy Test is to first simulate a covariate from a 1-d normal distribution. Then use the simulated values
    to generate the observations 2-d multivariate observations. This causes the y_i samples to come from distinct
    distributions whereas in first test after centering the y_i's are iid. The setup is identical to the first 
    simulation from "The Matrix-Logarithmic Covariance Model"
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[-5 - x_i, -3 + x_i],
              [-3 + x_i, -3 + x_i]]

    Target alpha is [-5, -1, -3, 1, -3, 1]
    """

    np.random.seed(1226789)

    X = np.random.normal(5, 1, 10)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "GradientDescent"
    initial_guess = 10 * np.random.rand(6)
    learning_rate = 0.03
    num_iterations = 10000
    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate": learning_rate,
                           "n_iter": num_iterations,
                           "tolerance": 1e-05}
    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    est_alpha = model.fit()

    for s in range(len(est_alpha) // 50 - 1):
        print('1st C Estimate: ' + str(s * 50))
        print(model.get_estimate(est_alpha[s * 50])[0])

    print('Final C Estimate: ')
    print(model.get_estimate(est_alpha[-1])[0])

    # Print Estimate for one sample for readability
    print("1st Target C:")
    print(C[0])

    print("Initial alpha guess:")
    print(initial_guess)
    print("Estimated alpha:")
    print(est_alpha[-1])
    print("Target alpha:")
    print([-5, -1, -3, 1, -3, 1])

if __name__ == "__main__":
    """
    Third Toy Test is to first simulate a covariate from a 1-d normal distribution. Then use the simulated values
    to generate the observations 2-d multivariate observations. This causes the y_i samples to come from distinct
    distributions whereas in first test after centering the y_i's are iid. The setup is identical to second toy test
    but ADMM is used rather than standard gradient descent.
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[-5 - x_i, -3 + x_i],
              [-3 + x_i, -3 + x_i]]

    Target alpha is [-5, -1, -3, 1, -3, 1]
    """

    np.random.seed(1226789)

    num_samples = 100
    X = np.random.normal(5, 1, num_samples)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "ADMM GD"
    initial_guess = 10 * np.random.rand(6)
    learning_rate = 0.03
    num_iterations = 100
    penalty_type = "L2"
    penalty_rate = 1/2
    max_iter = 5
    single_thread_optimizer_config = {"initial_guess": initial_guess,
                           "learning_rate": learning_rate,
                           "n_iter": num_iterations,
                           "tolerance": 1e-05}

    optimization_config = {"single_thread_optimizer_config": single_thread_optimizer_config,
                           "penalty_type": penalty_type,
                           "penalty_rate": penalty_rate,
                           "alpha_dim": 6,
                           "num_of_samples": num_samples,
                           "max_iter": max_iter,
                           "tolerance": 1e-05}

    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    print("Fitting Model")
    est_alpha = model.fit()

    print("Initial alpha guess:")
    print(initial_guess)
    print("Estimated alpha:")
    print(est_alpha)
    print("Target alpha:")
    print([-5, -1, -3, 1, -3, 1])


# Investigate Bregman Divergence
if __name__ == "__main__":
    """
    Fourth Toy Test is to for sample sizes [10, 50, 100, 250, 500, 1000, 2500, 
    5000] first simulate a covariate from a 1-d normal distribution. Then use 
    the simulated values to generate the observations 2-d multivariate 
    observations. This causes the y_i samples to come from distinct 
    distributions whereas in first test after centering the y_i's are iid. 
    The setup is identical to the first simulation from 
    "The Matrix-Logarithmic Covariance Model"
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[-5 - x_i, -3 + x_i],
              [-3 + x_i, -3 + x_i]]

    Target alpha is [-5, -1, -3, 1, -3, 1]
    We then compute the average Bregman Divergence for each of our final 
    estimates across every simulation size.
    """

    sim_samples = [10, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000]
    np.random.seed(1226789)
    simulated_alphas = {10: [], 50: [], 100: [], 250: [], 500: [], 1000: [],
                        2500: [], 5000: [], 7500: [], 10000: []}
    estimates = {10: [], 50: [], 100: [], 250: [], 500: [], 1000: [],
                 2500: [], 5000: [], 7500: [], 10000: []}
    sample_covariances = {10: [], 50: [], 100: [], 250: [], 500: [], 1000: [],
                          2500: [], 5000: [], 7500: [], 10000: []}
    target_values = {10: [], 50: [], 100: [], 250: [], 500: [], 1000: [],
                      2500: [], 5000: [], 7500: [], 10000: []}
    for sim_sample in sim_samples:
        print("Simulation with: " + str(sim_sample))
        X = np.random.normal(5, 1, sim_sample)
        A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

        # We can try a way of implementing the transformation that doesn't conflict
        # with factory design philosophy
        transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
        C = np.array([inverse_transform(a) for a in A])
        m = [0, 0]

        target_values[sim_sample] = C.copy()

        Y = [np.random.multivariate_normal(m, c) for c in C]

        sample_covs = [np.outer(y, y) for y in Y]

        # Construct the Entropic Covariance Model
        optimizer_type = "GradientDescent"
        initial_guess = 10 * np.random.rand(6)
        learning_rate = 0.003
        num_iterations = 2500
        optimization_config = {"initial_guess": initial_guess,
                               "learning_rate": learning_rate,
                               "n_iter": num_iterations,
                               "tolerance": 1e-05}
        model = EntropicCovModel("example_feature_map_2",
                                 "SMSI", X, Y,
                                 optimizer_type, optimization_config)

        est_alpha = model.fit()
        simulated_alphas[sim_sample] = est_alpha[-1]
        estimates[sim_sample] = model.get_estimate(est_alpha[-1])
        sample_covariances[sim_sample] = sample_covs

    print('Final Alpha Estimates')
    for sim_sample in sim_samples:
        print('Sample Size: ' + str(sim_sample))
        print('alpha: ' + str(simulated_alphas[sim_sample]))
    print("Target alpha:")
    print([-5, -1, -3, 1, -3, 1])

    print('Final Covariance Estimates')
    for sim_sample in sim_samples:
        print('Sample Size: ' + str(sim_sample))
        print('Estimate: ' + str(estimates[sim_sample][0]))

    print('Average Bregman Divergence from Estimate to Target Value')
    breg_div_target = []
    for sim_sample in sim_samples:
        bregman_divergences = []
        for i in range(sim_sample):
            bregman_divergences.append(model.compute_bregman_div(target_values[sim_sample][i], estimates[sim_sample][i]))
        avg_bregman = np.sum(bregman_divergences)/sim_sample
        breg_div_target.append(avg_bregman)

        print('Sample_Size: ' + str(sim_sample))
        print('Average Bregman Divergence: ' + str(avg_bregman))

    print(sim_samples)
    print(breg_div_target)
    labels = [str(sample) for sample in sim_samples]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.5
    plt.bar(labels, breg_div_target, color='skyblue', width=bar_width)

    # Add title and labels
    plt.title('Bregman Divergence from Estimate to True Covariance vs Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('Bregman Divergence to True Covariance')

    plt.savefig('divergence_to_target_vs_sample_labels.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.close()

if __name__ == "__main__":
    """
    Third Toy Test is to first simulate a covariate from a 1-d normal distribution. Then use the simulated values
    to generate the observations 2-d multivariate observations. This causes the y_i samples to come from distinct
    distributions whereas in first test after centering the y_i's are iid. The setup is identical to second toy test
    but ADMM is used rather than standard gradient descent.
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[-5 - x_i, -3 + x_i],
              [-3 + x_i, -3 + x_i]]

    Target alpha is [-5, -1, -3, 1, -3, 1]
    """

    np.random.seed(1226789)

    num_samples = 100
    X = np.random.normal(5, 1, num_samples)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "ADMM GD No Pen"
    initial_guess = 10 * np.random.rand(6)
    learning_rate = 0.03
    num_iterations = 100
    max_iter = 5
    single_thread_optimizer_config = {"initial_guess": initial_guess,
                                      "learning_rate": learning_rate,
                                      "n_iter": num_iterations,
                                      "tolerance": 1e-05}

    optimization_config = {"single_thread_optimizer_config": single_thread_optimizer_config,
                           "alpha_dim": 6,
                           "num_of_samples": num_samples,
                           "max_iter": max_iter,
                           "tolerance": 1e-05}

    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    print("Fitting Model")
    est_alpha = model.fit()

    print("Initial alpha guess:")
    print(initial_guess)
    print("Estimated alpha:")
    print(est_alpha)
    print("Target alpha:")
    print([-5, -1, -3, 1, -3, 1])

if __name__ == "1__main__":
    #TODO Gradient computation of log eventually becomes a complex number
    """
    Identical to the second toy test in terms of set up. We then compare the 
    performance of models using the SMSI link function and the log link funciton.
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[-5 - x_i, -3 + x_i],
              [-3 + x_i, -3 + x_i]]

    Target alpha is [-5, -1, -3, 1, -3, 1]
    """

    np.random.seed(1226789)

    num_samples = 100
    X = np.random.normal(5, 1, num_samples)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "GradientDescent"
    initial_guess = 10 * np.random.rand(6)
    learning_rate = 0.01
    num_iterations = 500
    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate": learning_rate,
                           "n_iter": num_iterations,
                           "tolerance": 1e-05}
    model_SMSI = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    print("Training SMSI")
    est_alpha_SMSI = model_SMSI.fit()

    model_log = EntropicCovModel("example_feature_map_2",
                                 "log", X, Y,
                                 optimizer_type, optimization_config)

    print("Training log")
    est_alpha_log = model_log.fit()

    print("True Paramters: [-5, -1, -3, 1, -3, 1]" )
    print("SMSI Estimates: " + str(est_alpha_SMSI[-1]))
    print("Log Estimates: " + str(est_alpha_log[-1]))


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

    num_samples = 5000
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
    num_iterations = 10000
    batch_size = 1

    # Determine whether batch is randomly sampled or cycles through data
    # Setting to True can be cumbersome for large sample sizes
    random_samples = False
    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate": learning_rate,
                           "n_iter": num_iterations,
                           "tolerance": 1e-05,
                           "batch_size": batch_size,
                           "random": random_samples,
                           "num_samples": num_samples}
    model = EntropicCovModel("example_feature_map_2",
                                  "SMSI", X, Y,
                                  optimizer_type, optimization_config)

    print("Training")
    est_alpha = model.fit()

    print("True Paramters: [-5, -1, -3, 1, -3, 1]" )
    print("Estimates: " + str(est_alpha[-1]))
