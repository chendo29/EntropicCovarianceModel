"""
    This is a sample code to conduct k-fold Cross Validation functionalities
"""
from EntropicCovModel import EntropicCovModel, LinkFunctionFactory
import numpy as np


if __name__ == "__main__":
    # Step 1: Simulate Data
    np.random.seed(122678)
    num_samples = 2500

    X = np.random.normal(25, 1, num_samples)
    A = np.array([[[5 * (x ** 2), 4 * x, -2 * x], [4 * x, x ** 2, -10 * x],
                   [-2 * x, -10 * x, 2 * (x ** 2)]] for x in X])

    # Step 2: Generate link function and its inverse
    transform, inverse_transform, _, _ = LinkFunctionFactory.create_links(
        "SMSI")

    # Step 3: Transform the data to generate Y
    C = np.array([inverse_transform(a) for a in A])
    m = [0, 0, 0]
    Y = [np.random.multivariate_normal(m, c) for c in C]

    # Step 4: Optimization Config for Entropic Cov Model
    optimizer_type = "StochasticGradientNewtonDescent"
    initial_guess = 50 * np.random.rand(6)
    learning_rate = 0.000001
    num_iterations_GD = 500
    num_iterations_newton = 2
    batch_size = 1

    optimization_config = {"initial_guess": initial_guess,
                           "learning_rate_GD": learning_rate,
                           "n_iter_GD": num_iterations_GD,
                           "n_iter_newton": num_iterations_newton,
                           "tolerance_gd": 1e-10,
                           "tolerance": 1e-15,
                           "batch_size": batch_size,
                           "num_samples": None}

    # Step 5: k-fold CV functionalities
    def mean_squared_error(cov_true, cov_pred):
        return np.mean((cov_true - cov_pred) ** 2)

    def frobenius_norm(A, B):
        return np.linalg.norm(A - B, ord='fro')

    def operator_norm(A, B):
        return np.linalg.norm(A - B, ord=2)

    def l_inf_norm(A, B):
        return np.linalg.norm(A - B, ord=np.inf)

    def k_fold_cross_validation(X, y, k, error_measure="MSE"):
        n = len(X)
        fold_size = n // k
        mean_error_scores = []

        for fold in range(k):
            # Split data into training and validation sets
            X_val = X[fold * fold_size:(fold + 1) * fold_size]
            Y_val = y[fold * fold_size:(fold + 1) * fold_size]
            C_val = C[fold * fold_size:(fold + 1) * fold_size]
            X_train = np.concatenate(
                (X[:fold * fold_size], X[(fold + 1) * fold_size:]))
            Y_train = Y[:fold * fold_size] + Y[(fold + 1) * fold_size:]

            # Update the config
            optimization_config['num_samples'] = len(X_train)
            # Train the model
            model = EntropicCovModel("example_feature_map_5",
                                     "SMSI", X_train, Y_train,
                                     optimizer_type, optimization_config)
            alphas = model.fit(save_path=False)
            C_pred = model.predict(X_val, alphas)

            # Compute Prediction Error
            if error_measure == "MSE":
                error = mean_squared_error(C_val, C_pred)
            elif error_measure == "Frobenius_norm":
                error = np.mean([frobenius_norm(c_pred, c_val) for
                                 c_pred, c_val in zip(C_pred, C_val)])
            elif error_measure == "Operator_norm":
                error = np.mean([operator_norm(c_pred, c_val) for
                                 c_pred, c_val in zip(C_pred, C_val)])
            elif error_measure == "L_inf_norm":
                error = np.mean([l_inf_norm(c_pred, c_val) for
                                 c_pred, c_val in zip(C_pred, C_val)])
            else:
                raise ValueError("Invalid error measure")
            mean_error_scores.append(error)

        return np.array(mean_error_scores)


    # Perform 5-fold cross-validation
    k = 5
    cv_scores = k_fold_cross_validation(X, Y, k, "Operator_norm")

    # Output the results
    print(f"Cross-Validation Mean Square Errors: {cv_scores}")
    print(f"Average Mean Square Errors: {np.mean(cv_scores)}")
    print(f"Standard Deviation of Mean Square Errors: {np.std(cv_scores)}")
