"""

This module contains an example on the usage of Entropic Covariance Model.

"""
import numpy as np
import time

from EntropicCovModel import EntropicCovModel, LinkFunctionFactory

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
    num_samples = 10

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
    learning_rate = 0.03
    num_iterations = 10

    optimization_config = {"initial_guess": initial_guess,
                                    "learning_rate": learning_rate,
                                    "n_iter": num_iterations,
                                    "tolerance": 1e-05,}
    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    t = time.time()
    est_alpha = model.fit()
    print(time.time()-t)

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
