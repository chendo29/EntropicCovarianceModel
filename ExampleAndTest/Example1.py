"""

This module contains an example on the usage of Entropic Covariance Model.

"""
import numpy as np
from EntropicCovModel import EntropicCovModel


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
