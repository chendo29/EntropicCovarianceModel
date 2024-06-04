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


class FeatureMapBase(ABC):
    """
    Define an abstract base class as an interface for a feature mapping. This allows
    flexible extensions for the choice of maps taking design matrix X into basis for
    the linear subspace of symmetric matrices.
    """

    @abstractmethod
    def __call__(self, x):
        pass


class ExampleFeatureMap(FeatureMapBase):
    # Feature map compatible with toy example
    # TODO: This is basis will cause Singular matrix if the initial guess is 1
    def __call__(self, x):
        u1 = np.array([[x[0], 0],
                       [0, 0]])
        u2 = np.array([[0, 0],
                       [0, x[0]]])
        u3 = np.array([[x[1], 0],
                       [0, 0]])
        u4 = np.array([[0, 0],
                       [0, x[1]]])
        u5 = np.array([[1, 0],
                       [0, 0]])
        u6 = np.array([[0, 0],
                       [0, 1]])
        u7 = np.array([[0, 1],
                       [1, 0]])
        u8 = np.array([[0, x[0]],
                       [x[0], 0]])
        u9 = np.array([[0, x[1]],
                       [x[1], 0]])
        basis = [u1, u2, u3, u4, u5, u6, u7, u8, u9]
        return basis


class FeatureMapFactory:
    # Define a feature map factory to produce concrete feature maps
    @staticmethod
    def create_feature_maps(map_type):
        if map_type == "example_feature_map":
            return ExampleFeatureMap()
        else:
            raise ValueError(f"Unknown feature map type: {map_type}")


class SubspaceBases:
    """
    Store list of bases for each linear subspace over which optimization
    will occur. There is exactly one basis for every sample, each of which is
    constructed with the same feature mapping.
    """

    def __init__(self, feature_map_type, design_matrices):
        # Element i represents [U_i1,...,U_il] in JASA paper
        feature_map = FeatureMapFactory.create_feature_maps(feature_map_type)
        self.bases = [feature_map(xi) for xi in design_matrices]

    def get_subspace_basis(self):
        return self.bases


class EntropicCovModel:
    """
    This is a class that holding the information with the following field
    1. The link function (nabla F)
    2. The inverse of link function (nabla F^*)
    3. The design matrix X
    4. The response vector Y
    """

    def __init__(self, feature_map_type, link_type, design_matrix,
                 response_vector):
        self.link_func, self.inverse_link_func \
            = LinkFunctionFactory.create_links(link_type)
        self.Y = response_vector
        self.bases = SubspaceBases(feature_map_type, design_matrix).get_subspace_basis()

    def apply_link_func(self, mat):
        return self.link_func(mat)

    def apply_inverse_link_func(self, mat):
        return self.inverse_link_func(mat)

    def compute_gradient(self, alpha):
        """
        This method computes the gradient our loss function evaluated at point
        alpha. The loss function is defined to be the sum of Bregman divergence.
        :param alpha: the coefficient of the regression model applied on
        link-function transformation of the covariance matrix
        :return: the gradient of the loss function at point alpha. Equivalently
        a vector lives in R^l, where l is the number of features.
        """
        A_alpha = self._compute_A_alpha(alpha)
        gradient = np.zeros_like(alpha)
        for i in range(len(self.Y)):
            A_i_alpha = A_alpha[i]
            y_i = self.Y[i]
            M = self.apply_link_func(A_i_alpha) - np.dot(y_i, y_i)
            U = self.bases[i]
            gradient += self._adjoint_map(U, M)
        return gradient

    def optimize(self, initial_guess, learning_rate,
                 num_iterations, tol=1e-06):
        return Optimizer.gradient_descent(self.compute_gradient, initial_guess,
                                          learning_rate, num_iterations, tol)

    """
    The followings are private helper method to be used in the body of 
    compute_gradient
    """
    def _compute_A_alpha(self, alpha):
        # Private helper method to compute A_i(alpha) for all i
        A_alpha = []
        for basis in self.bases:
            weighted_matrices = [weight * mat for weight, mat in zip(alpha, basis)]
            A_alpha.append(np.sum(weighted_matrices, axis=0))
        return A_alpha

    def _adjoint_map(self, basis_list, m):
        # Private helper method to compute the adjoint A^* with some input
        # matrix m
        vec = []
        for u in basis_list:
            mat = u @ m
            vec.append(np.trace(mat))
        return vec


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
    model = EntropicCovModel("example_feature_map", "SMSI", X, Y)
    initial_guess = 100 * np.random.rand(9)
    learning_rate = 0.01
    num_iterations = 100
    est_C = model.optimize(initial_guess, learning_rate, num_iterations)
    print(est_C)
    a = 0
    # Example usage
    # def execute_function(func: LinkFunctionBase, *args, **kwargs):
    #    return func(*args, **kwargs)
