import numpy as np

from EntropicCovModel import EntropicCovModel, LinkFunctionFactory

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
