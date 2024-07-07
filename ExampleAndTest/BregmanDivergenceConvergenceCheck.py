"""
This test is to show convergence of the EntropicCovModel
for sample sizes [10, 50, 100, 250, 500, 1000, 2500,
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
import numpy as np
from matplotlib import pyplot as plt

from EntropicCovModel import EntropicCovModel, LinkFunctionFactory

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

    sim_samples = [1000, 5000, 10000, 20000, 30000]
    np.random.seed(1226789)
    simulated_alphas = {}
    estimates = {}
    sample_covariances = {}
    target_values = {}
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

        optimizer_type = "SGD"
        initial_guess = 10 * np.random.rand(6)
        learning_rate = 0.0005
        num_iterations = 10000
        batch_size = 250

        optimization_config = {"initial_guess": initial_guess,
                               "learning_rate": learning_rate,
                               "n_iter": num_iterations,
                               "tolerance": 1e-05,
                               "batch_size": batch_size,
                               "num_samples": sim_sample}
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

    plt.savefig('divergence_to_target_vs_sample_labels_30k_more_its.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.close()

