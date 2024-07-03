"""

This module contains demos for plotting landscape of loss around optimum.

"""
import numpy as np
import matplotlib.pyplot as plt
from EntropicCovModel import EntropicCovModel, LinkFunctionFactory


if __name__ == "__main__":
    """
    Generate guesses and plots loss of interpolated points between guess and
    true value
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[-5 - x_i, -3 + x_i],
              [-3 + x_i, -3 + x_i]]

    Target alpha is [-5, -1, -3, 1, -3, 1]
    """

    np.random.seed(12345)
    num_samples = 1

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
    num_iterations = 250

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate": learning_rate,
                           "n_iter": num_iterations,
                           "tolerance": 1e-10,}
    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    true_value = np.array([-5, -1, -3, 1, -3, 1])

    # Interpolates between initial guess and true covariance
    interp_losses = []
    num_steps = 250
    interp_steps = list(range(num_steps*4))
    sample_cov = np.outer(Y[0], Y[0])
    true_value_mat = model.get_estimate(true_value)[0]

    num_guesses = 8
    guesses = [10 * np.random.rand(6) for _ in range(num_guesses)]
    interp_loss_list = []
    true_value_div = []

    for guess in guesses:
        interp_losses = []
        true_value_mat = model.get_estimate(true_value)[0]
        true_value_div.append(model.compute_bregman_div_no_samp(sample_cov, true_value_mat))
        for t in interp_steps:
            interp = (1 - t/num_steps)*guess + (t/num_steps)*true_value
            interp_mat = model.get_estimate(interp)[0]
            interp_losses.append(model.compute_bregman_div_no_samp(sample_cov, interp_mat))
        print(interp_losses[0])
        interp_loss_list.append(interp_losses)

    fig, axs = plt.subplots(num_guesses//2, 2, figsize=(12, 8))

    counter = 0
    for ax in axs.flat:
        if counter < num_guesses:
            ax.plot(interp_steps, interp_loss_list[counter])
            ax.set(xlabel='Steps', ylabel='Loss')
            min_index = interp_loss_list[counter].index(min(interp_loss_list[counter]))
            print('True Value Divergence: ' + str(true_value_div[counter]))
            print('Graph Minimum: ' + str(interp_loss_list[counter][min_index]))
            ax.axvline(x=interp_steps[num_steps], color='r', linestyle='--')
            counter += 1

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()

if __name__ == "__main__":
    """
    Plots loss of estimated covariance at each step of Gradient Descent
        i) x_i ~ N(5, 1)
        ii) y_i ~ MVN(0, C(x_i))
    Where the covariance is given as a function of x_i
    C(x_i) = apply_inverse_link_func(A(x_i))
    A(x_i) = [[-5 - x_i, -3 + x_i],
              [-3 + x_i, -3 + x_i]]

    Target alpha is [-5, -1, -3, 1, -3, 1]
    """

    np.random.seed(1226789)
    num_samples = 1

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
    num_iterations = 500

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate": learning_rate,
                           "n_iter": num_iterations,
                           "tolerance": 1e-10,}
    model = EntropicCovModel("example_feature_map_2",
                             "SMSI", X, Y,
                             optimizer_type, optimization_config)

    est_alpha = model.fit()

    true_value = np.array([-5, -1, -3, 1, -3, 1])

    sample_cov = np.outer(Y[0], Y[0])

    model_losses = []
    for alpha in est_alpha:
        estimate = model.get_estimate(alpha)[0]
        model_losses.append(model.compute_bregman_div_no_samp(sample_cov,
                                                              estimate))

    iterations = list(range(len(est_alpha)))

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.5
    plt.bar(iterations, model_losses, color='skyblue', width=bar_width)

    # Add title and labels
    plt.title('Loss Landscape at Each Step')
    plt.ylabel('Bregman Divergence')

    plt.show()
