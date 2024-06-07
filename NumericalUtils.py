"""

This is a python module to provide Numerical utilities to support the Entropic
Covariance Model. This includes various optimization routine, numerical
differentiations etc

Author: Dongli Chen, Department of Statistics, University of Toronto, dongli.chen@mail.utoronto.ca
        William Groff, Department of Statistics, University of Toronto, william.groff@mail.utoronto.ca
        Piotr Zwiernik, Department of Statistics, University of Toronto, piotr.zwiernik@utoronto.ca
"""

import numpy as np


class Optimizer:
    @staticmethod
    def gradient_descent(gradient, start, learn_rate, n_iter, tolerance=1e-05):
        """
        Performs gradient descent optimization.

        Parameters:
        - gradient: Function to compute the gradient of the function to be minimized.
        - start: Initial starting point for the algorithm.
        - learn_rate: Learning rate (step size) for each iteration.
        - n_iter: Number of iterations to perform.
        - tolerance: Tolerance for stopping criteria.

        Returns:
        - vector of positions for each iteration
        """
        x = start
        positions = [x]

        for _ in range(n_iter):
            grad = gradient(x)
            x_new = x - learn_rate * grad
            positions.append(x_new)

            # Stop if the change is smaller than the tolerance
            if np.all(np.abs(x_new - x) <= tolerance):
                break
            x = x_new

        return np.array(positions)

    @staticmethod
    # TODO - William: Implement Bregman Projection Optimization Project
    def bregman_projection():
        pass

