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
from scipy.linalg import inv

from FeatureMapping import FeatureMapFactory
from NumericalUtils import OptimizerFactory


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


class FunctionBase(ABC):
    """
    Define an abstract base class as an interface of the base function.
    This allows flexible extensions on the family of base functions in the
    future since one can implement a new concrete child class
    for a new base function.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SMSIBaseFunction(FunctionBase):
    # Base function that induces the SMSI link function
    def __call__(self, mat):
        return 1/2*(np.trace(mat @ mat)) - np.log(np.linalg.det(mat))


class LogBaseFunction(FunctionBase):
    # Base function that induces the log base function
    def __call__(self, mat):
        return -1*np.trace(mat - mat@logm(mat))


class SMSIBaseConjugate(FunctionBase):
    # Conjugate of Base Function that induces SMSI link function
    def __call__(self, mat):
        num_rows = mat.shape[0]
        mat_bar = mat @ mat + 4 * np.eye(num_rows)
        mat2 = 0.5 * (mat + sqrtm(mat_bar))
        return np.trace(mat@mat2 - 0.5*mat2@mat2 + logm(mat2))


class LogBaseConjugate(FunctionBase):
    # Conjugate of Base Function that induces log link function
    def __call__(self, mat):
        return np.trace(expm(mat))


class LinkFunctionFactory:
    # Define a link function factory to produce link functions and its inverse
    @staticmethod
    def create_links(link_type):
        if link_type == "log":
            return Log(), InverseLog(), LogBaseFunction(), LogBaseConjugate()
        elif link_type == "SMSI":
            return SMSI(), InverseSMSI(), SMSIBaseFunction(), SMSIBaseConjugate()
        else:
            raise ValueError(f"Unknown link type: {link_type}")


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
    5. The optimizer
    """

    def __init__(self, feature_map_type, link_type, design_matrix,
                 response_vector, optimizer_type, optimization_conifg):
        """

        :param feature_map_type: type of the feature map
        :param link_type: type of the link function
        :param design_matrix: X
        :param response_vector: Y
        :param optimizer_type: type of the optimizer
        :param optimization_conifg: configuration for the optimizer to conduct
        its optimization routine
        """
        self.link_func, self.inverse_link_func, self.base_func, \
        self.base_func_conjugate = LinkFunctionFactory.create_links(link_type)
        self.Y = response_vector
        self.feature_map_type = feature_map_type
        self.bases = SubspaceBases(feature_map_type, design_matrix).get_subspace_basis()
        self.optimizer = OptimizerFactory.create_optimizer(optimizer_type,
                                                           optimization_conifg,
                                                           self)

    def apply_link_func(self, mat):
        return self.link_func(mat)

    def apply_inverse_link_func(self, mat):
        return self.inverse_link_func(mat)

    def compute_bregman_div(self, mat1, mat2):
        # Compute Bregman
        return self.base_func(mat1) + self.base_func_conjugate(mat2) - np.trace(mat1@mat2)

    def compute_bregman_div_no_samp(self, mat1, mat2):
        # Compute Bregman without first term in case mat1 is not PSD
        return self.base_func_conjugate(mat2) - np.trace(mat1@mat2)

    def compute_hessian(self, alpha):
        # Hessian associated with SMSI map

        dim = len(self.bases[0])
        num_samples = len(self.bases)
        hessian = np.zeros((dim,dim))
        A_alphas = self._compute_A_alpha(alpha)

        for k in range(dim):
            for j in range(dim):
                summands = []
                for i in range(num_samples):
                    Uik = self.bases[i][k]
                    Uij = self.bases[i][j]
                    D = inv(sqrtm(A_alphas[i]@A_alphas[i] + 4*np.eye(len(A_alphas[0]))))
                    summands.append(np.trace(Uik@Uij + Uik@D@Uij))
                hessian[j][k] = np.sum(summands)
        return hessian

    def compute_hessian_diag(self, alpha):
        # Hessian associated with SMSI map

        dim = len(self.bases[0])
        num_samples = len(self.bases)
        hessian = np.zeros((dim,dim))
        A_alphas = self._compute_A_alpha(alpha)

        for j in range(dim):
            summands = []
            for i in range(num_samples):
                Uij = self.bases[i][j]
                D = inv(sqrtm(A_alphas[i]@A_alphas[i] + 4*np.eye(len(A_alphas[0]))))
                summands.append(np.trace(Uij@Uij + Uij@D@Uij))
            hessian[j][j] = np.sum(summands)
        return hessian

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
            M = self.apply_link_func(A_i_alpha) - np.outer(y_i, y_i)
            U = self.bases[i]
            gradient += self._adjoint_map(U, M)
        return gradient

    def compute_batch_gradient(self, alpha, batch_indices):
        """
        This method computes the gradient of each term of our loss function
        evaluated at point alpha. The loss function is defined to be the sum
        of Bregman divergence. This method computes the gradient of each term
        of our loss function associated to the elements of our batch evaluated
        at point alpha. The loss function is defined to be the sum of Bregman
        divergence.
        :param alpha: the coefficient of the regression model applied on
        link-function transformation of the covariance matrix
        :param batch_indices: the indices of the batch for stochastic gradient
        descent
        :return: the gradient of the loss function at point alpha. Equivalently
        a vector lives in R^l, where l is the number of features.
        """
        A_alpha = self._compute_A_alpha(alpha)
        gradient = np.zeros_like(alpha)

        for i in batch_indices:
            A_i_alpha = A_alpha[i]
            y_i = self.Y[i]
            M = self.apply_link_func(A_i_alpha) - np.outer(y_i, y_i)
            U = self.bases[i]
            gradient += self._adjoint_map(U, M)
        gradient = gradient
        return gradient

    def fit(self):
        return self.optimizer.optimize()

    def get_estimate(self, alpha):
        A_alpha = self._compute_A_alpha(alpha)
        estimates = [self.inverse_link_func(A_i) for A_i in A_alpha]

        return estimates

    def get_A_alpha(self, alpha):
        return self._compute_A_alpha(alpha)

    def predict(self, X, alphas):
        predicted_bases = SubspaceBases(self.feature_map_type, X).get_subspace_basis()
        predicted_matrices = []
        for base in predicted_bases:
            transform_mat = np.sum([a * u for a, u in zip(alphas, base)])
            inverse_transform_mat = self.inverse_link_func(transform_mat)
            predicted_matrices.append(inverse_transform_mat)
        return predicted_matrices

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




