"""

This module contains an example on the usage of Entropic Covariance Model.

"""
import numpy as np

from EntropicCovModel import EntropicCovModel, LinkFunctionFactory

if __name__ == "__main__":
    # TODO Gradient computation of log eventually becomes a complex number
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
