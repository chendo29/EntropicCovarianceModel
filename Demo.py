"""

This module contains demos on the usage of Entropic Covariance Model.

"""
import numpy as np

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

    X = np.random.normal(5, 1, 1500)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform = LinkFunctionFactory.create_links("SMSI")
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
                           "num_of_iteration": num_iterations,
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
    # TODO: provide examples to debug ADMM approach
