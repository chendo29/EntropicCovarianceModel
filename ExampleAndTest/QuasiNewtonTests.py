"""

This module contains an example on the usage of Entropic Covariance Model.

"""
import numpy as np
import time

from EntropicCovModel import EntropicCovModel, LinkFunctionFactory


if __name__ == "__main__":
    """
    Test Convergence of Newton Descent Optimizer is to first simulate a covariate from a 1-d normal distribution. Then use the simulated values
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
    num_samples = 1000

    X = np.random.normal(5, 1, num_samples)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "GradientNewtonDescent"
    initial_guess = 10 * np.random.rand(6)
    learning_rate = 0.000005
    num_iterations_GD = 1000
    num_iterations_newton = 100

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate_GD": learning_rate,
                           "n_iter_GD": num_iterations_GD,
                           "n_iter_newton": num_iterations_newton,
                           "tolerance_gd": 1e-10,
                           "tolerance": 1e-15}
    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    t = time.time()
    print("Initial Guess: " + str(initial_guess))
    est_alpha = model.fit()

    # Print Estimate for one sample for readability
    print("1st Target C:")
    print(C[0])
    print('Final C Estimate: ')
    print(model.get_estimate(est_alpha[-1])[0])

    print("Initial alpha guess:")
    print(initial_guess)
    print("Estimated alpha:")
    print(est_alpha[-1])
    print("Target alpha:")
    print([-5, -1, -3, 1, -3, 1])


if __name__ == "__main__":
    """
    Test Convergence of Stochastic Newton Descent Optimizer is to first simulate a covariate from a 1-d normal distribution. Then use the simulated values
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
    num_samples = 10000

    X = np.random.normal(5, 1, num_samples)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "StochasticGradientNewtonDescent"
    initial_guess = 10 * np.random.rand(6)
    learning_rate = 0.005
    num_iterations_GD = 50000
    num_iterations_newton = 5
    batch_size = 1

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate_GD": learning_rate,
                           "n_iter_GD": num_iterations_GD,
                           "n_iter_newton": num_iterations_newton,
                           "tolerance_gd": 1e-15,
                           "tolerance": 0,
                           "batch_size": batch_size,
                           "num_samples": num_samples}

    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    t = time.time()
    est_alpha = model.fit()

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
    Test Convergence of Stochastic Newton Descent Optimizer is to first simulate a covariate from a 1-d normal distribution. Then use the simulated values
    to generate the observations 2-d multivariate observations. This causes the y_i samples to come from distinct
    distributions whereas in first test after centering the y_i's are iid. The setup is identical to the first 
    simulation from "The Matrix-Logarithmic Covariance Model"
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[- x_i, x_i],
              [x_i, x_i]]

    Target alpha is [1]
    """

    np.random.seed(122678)
    num_samples = 1000

    X = np.random.normal(25, 1, num_samples)
    A = np.array([[[- x, x], [x, x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "StochasticGradientNewtonDescent"
    initial_guess = 25 * np.random.rand(1)
    learning_rate = 0.00005
    num_iterations_GD = 1000
    num_iterations_newton = 3
    batch_size = 1

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate_GD": learning_rate,
                           "n_iter_GD": num_iterations_GD,
                           "n_iter_newton": num_iterations_newton,
                           "tolerance_gd": 1e-15,
                           "tolerance": 0,
                           "batch_size": batch_size,
                           "num_samples": num_samples}

    model = EntropicCovModel("example_feature_map_3",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    t = time.time()
    est_alpha = model.fit()

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
    print([1])


if __name__ == "__main__":
    """
    Test Convergence of Newton's Method Optimizer is to first simulate a covariate from a 1-d normal distribution. Then use the simulated values
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

    np.random.seed(122678)
    num_samples = 250

    X = np.random.normal(5, 1, num_samples)
    A = np.array([[[-5 - x, -3 + x], [-3 + x, -3 + x]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "NewtonMethod"
    initial_guess = 20 * np.random.rand(6)
    num_iterations = 50

    optimization_config = {"initial_guess": initial_guess,
                           "n_iter": num_iterations,
                           "tolerance": 0}

    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    t = time.time()
    print('Fitting Model')
    est_alpha = model.fit()

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
    Test Convergence of Stochastic Newton Descent Optimizer on feature maps with non linear covariate terms is to first
    simulate a covariate from a 1-d normal distribution. Then use the simulated values
    to generate the observations 2-d multivariate observations. This causes the y_i samples to come from distinct
    distributions whereas in first test after centering the y_i's are iid. The setup is identical to the first 
    simulation from "The Matrix-Logarithmic Covariance Model"
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[5(x_i)**2, 2(x_i)**2],
              [2(x_i)**2, 5(x_i)**2]]

    Target alpha is [5, 2, 5]
    """

    np.random.seed(122678)
    num_samples = 50

    X = np.random.normal(15, 1, num_samples)
    A = np.array([[[5*(x**2), 2*(x**2)], [2*(x**2), 5*(x**2)]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "StochasticGradientNewtonDescent"
    initial_guess = 10 * np.random.rand(3)
    learning_rate = 0.00002
    num_iterations_GD = 1000
    num_iterations_newton = 1
    batch_size = 1

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate_GD": learning_rate,
                           "n_iter_GD": num_iterations_GD,
                           "n_iter_newton": num_iterations_newton,
                           "tolerance_gd": 1e-15,
                           "tolerance": 0,
                           "batch_size": batch_size,
                           "num_samples": num_samples}

    model = EntropicCovModel("example_feature_map_4",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    t = time.time()
    print('Fitting Model')
    est_alpha = model.fit()

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
    print([5, 2, 5])


if __name__ == "__main__":
    """
    Test Convergence of Stochastic Newton Descent Optimizer on feature maps with non linear covariate terms is to first
    simulate a covariate from a 1-d normal distribution. Then use the simulated values
    to generate the observations 2-d multivariate observations. This causes the y_i samples to come from distinct
    distributions whereas in first test after centering the y_i's are iid. The setup is identical to the first 
    simulation from "The Matrix-Logarithmic Covariance Model"
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))      
    A(x_i) = [[5(x_i)**2, 4x_i, 2x_i], 
              [4x_i, (x_i)**2, -10x_i], 
              [2x_i, -10x_i, 2(x_i)**2]]

    Target alpha is [5, 1, 2, 4, -10, -2]
    """

    np.random.seed(122678)
    num_samples = 2000

    X = np.random.normal(25, 1, num_samples)
    A = np.array([[[5*(x**2), 4*x, -2*x],
                   [4*x, x**2, -10*x],
                   [-2*x, -10*x, 2*(x**2)]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "StochasticGradientNewtonDescent"
    initial_guess = 500 * np.random.rand(6)
    learning_rate = 0.000005
    num_iterations_GD = 2000
    num_iterations_newton = 3
    batch_size = 1

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate_GD": learning_rate,
                           "n_iter_GD": num_iterations_GD,
                           "n_iter_newton": num_iterations_newton,
                           "tolerance_gd": 1e-15,
                           "tolerance": 0,
                           "batch_size": batch_size,
                           "num_samples": num_samples}

    model = EntropicCovModel("example_feature_map_5",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    t = time.time()
    print('Fitting Model')
    est_alpha = model.fit()

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
    print([5, 1, 2, 4, -10, -2])


if __name__ == "__main__":
    """
    Test Convergence of Stochastic Newton Descent Optimizer on feature maps with non linear covariate terms is to first
    simulate a covariate from a 1-d normal distribution. Then use the simulated values
    to generate the observations 2-d multivariate observations. This causes the y_i samples to come from distinct
    distributions whereas in first test after centering the y_i's are iid. The setup is identical to the first 
    simulation from "The Matrix-Logarithmic Covariance Model"
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))      
    A(x_i) = [[5(x_i)**2, 4x_i, 2x_i], 
              [4x_i, (x_i)**2, -10x_i], 
              [2x_i, -10x_i, 2(x_i)**2]]

    Target alpha is [5, 1, 2, 4, -10, -2]
    """

    np.random.seed(122678)
    num_samples = 250

    X = np.random.normal(25, 1, num_samples)
    A = np.array([[[5 * (x ** 2), 4 * x, -2 * x],
                   [4 * x, x ** 2, -10 * x],
                   [-2 * x, -10 * x, 2 * (x ** 2)]] for x in X])

    # We can try a way of implementing the transformation that doesn't conflict
    # with factory design philosophy
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links("SMSI")
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0, 0]

    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Construct the Entropic Covariance Model
    optimizer_type = "StochasticGradientQuasiNewtonDescent"
    initial_guess = 500 * np.random.rand(6)
    learning_rate = 0.000005
    num_iterations_GD = 1500
    num_iterations_newton = 3
    batch_size = 1

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate_GD": learning_rate,
                           "n_iter_GD": num_iterations_GD,
                           "n_iter_newton": num_iterations_newton,
                           "tolerance_gd": 1e-15,
                           "tolerance": 0,
                           "batch_size": batch_size,
                           "num_samples": num_samples}

    model = EntropicCovModel("example_feature_map_5",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    t = time.time()
    print('Fitting Model')
    est_alpha = model.fit()

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
    print([5, 1, 2, 4, -10, -2])


