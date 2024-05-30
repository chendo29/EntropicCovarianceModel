"""

This is a python module to implement the Entropic Covariance Model using
Bergman Divergence Estimator. The model leverage on the modeling approach
propose in https://arxiv.org/pdf/2306.03590 to build a GLM framework to
modeling the covariance matrix structure

Author: Dongli Chen, Department of Statistics, University of Toronto, dongli.chen@mail.utoronto.ca
        William Groff, Department of Statistics, University of Toronto, william.groff@mail.utoronto.ca
        Piotr Zwiernik, Department of Statistics, University of Toronto, piotr.zwiernik@utoronto.ca

"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import logm
from scipy.linalg import expm
from scipy.linalg import sqrtm


class Optimizer:
    # TODO - William: Implement Bregman Projection Optimization Project
    @staticmethod
    def gradient_descent(gradient, start, learn_rate, n_iter, tolerance=1e-06):
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


class LinkFunctionBase(ABC):
    """
    Define an abstract base class as an interface of a link function.
    This allows flexible extensions on the family of link functions in the
    future since one can implement a new concrete child class
    for a new link function
    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Log(LinkFunctionBase):
    # Log link
    def __call__(self, mat):
        return logm(mat)


class InverseLog(LinkFunctionBase):
    # Inverse of the log link
    def __call__(self, mat):
        return expm(mat)


class SMSI(LinkFunctionBase):
    # Sigma - Sigma^{-1} link
    def __call__(self, mat):
        return mat - np.linalg.inv(mat)


class InverseSMSI(LinkFunctionBase):
    # Inverse of Sigma - Sigma^{-1} link
    def __call__(self, mat):
        num_rows = mat.shape[0]
        mat_bar = mat @ mat + 4 * np.eye(num_rows)
        return 0.5 * (mat + sqrtm(mat_bar))


class LinkFunctionFactory:
    # Define a link function factory to produce link functions and its inverse
    @staticmethod
    def create_links(link_type):
        if link_type == "log":
            return Log(), InverseLog()
        elif link_type == "SMSI":
            return SMSI(), InverseSMSI()
        else:
            raise ValueError(f"Unknown link type: {link_type}")


class AbstractSubspaceBasis(ABC):
    """
    Define an abstract base class as an interface of a basis. This allows
    flexible extensions on the choice of basis for the (affine) linear subspace
    TODO - William: Implement the Basis/Feature Maps
    """
    @abstractmethod
    def get_subspace_basis(self):
        pass


class EntropicCovModel:
    """
    This is a class that holding the information with the following field
    1. The link function (nabala F)
    2. The inverse of link function (nabala F^*)
    3. The design matrix X
    4. The response vector Y
    """
    def __init__(self, link_type, design_matrix, response_vector):
        self.link_func, self.inverse_link_func = LinkFunctionFactory.create_links(link_type)
        self.X = design_matrix
        self.Y = response_vector

    def apply_link_func(self, mat):
        return self.link_func(mat)

    def apply_inverse_link_func(self, mat):
        return self.inverse_link_func(mat)

    def compute_gradient(self):
        # TODO: Complete this function after Tuesday's meeting
        pass

    def optimize(self, initial_guess, learning_rate,
                 num_iterations, tol=1e-06):
        # TODO: Complete this function after Tuesday's meeting
        pass


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

    print("Generated Samples:")
    print(samples)
    print("Generated X:")
    print(X)
    print("Generated Y:")
    print(Y)

    # Construct the Entropic Covariance Model
    model = EntropicCovModel("SMSI", X, Y)
    # TODO: est_C = model.optimize()
    # Example usage
    # def execute_function(func: LinkFunctionBase, *args, **kwargs):
    #    return func(*args, **kwargs)

