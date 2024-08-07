"""

This is the feature mapping module to provide feature maps for the Entropic
Covariance Model.

Author: Dongli Chen, Department of Statistics, University of Toronto, dongli.chen@mail.utoronto.ca
        William Groff, Department of Statistics, University of Toronto, william.groff@mail.utoronto.ca
        Piotr Zwiernik, Department of Statistics, University of Toronto, piotr.zwiernik@utoronto.ca

"""

from abc import ABC, abstractmethod

import numpy as np


class FeatureMapBase(ABC):
    """
    Define an abstract base class as an interface for a feature mapping. This allows
    flexible extensions for the choice of maps taking design matrix X into basis for
    the linear subspace of symmetric matrices.
    """

    @abstractmethod
    def __call__(self, x):
        pass


class ExampleFeatureMap1(FeatureMapBase):
    # Feature map compatible with toy example
    # Warning: This basis will cause Singular matrix if the initial guess is 1
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


class ExampleFeatureMap2(FeatureMapBase):
    # Feature map compatible with second toy example
    def __call__(self, x):
        u1 = np.array([[1, 0],
                       [0, 0]])
        u2 = np.array([[x, 0],
                       [0, 0]])
        u3 = np.array([[0, 0],
                       [0, 1]])
        u4 = np.array([[0, 0],
                       [0, x]])
        u5 = np.array([[0, 1],
                       [1, 0]])
        u6 = np.array([[0, x],
                       [x, 0]])
        basis = [u1, u2, u3, u4, u5, u6]
        return basis


class ExampleFeatureMap3(FeatureMapBase):
    # Feature map with linearly independent terms
    def __call__(self, x):
        u1 = np.array([[-x, x],
                       [x, x]])
        basis = [u1]
        return basis


class ExampleFeatureMap4(FeatureMapBase):
    # Feature map with nonlinear x terms
    def __call__(self, x):
        u1 = np.array([[x**2, 0],
                      [0, 0]])
        u2 = np.array([[0, x**2],
                      [x**2, 0]])
        u3 = np.array([[0, 0],
                      [0, x**2]])
        basis = [u1, u2, u3]
        return basis


class ExampleFeatureMap5(FeatureMapBase):
    # Feature map with nonlinear x terms
    def __call__(self, x):
        u1 = np.array([[x**2, 0, 0],
                      [0, 0, 0],
                       [0, 0, 0]])

        u2 = np.array([[0, 0, 0],
                       [0, x**2, 0],
                       [0, 0, 0]])

        u3 = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, x**2]])

        u4 = np.array([[0, x, 0],
                       [x, 0, 0],
                       [0, 0, 0]])

        u5 = np.array([[0, 0, 0],
                       [0, 0, x],
                       [0, x, 0]])

        u6 = np.array([[0, 0, x],
                       [0, 0, 0],
                       [x, 0, 0]])

        basis = [u1, u2, u3, u4, u5, u6]
        return basis


class ExampleFeatureMap6(FeatureMapBase):
    # Feature map with nonlinear x terms
    def __call__(self, x):
        u1 = np.array([[x**3, 0, 0, 0],
                      [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])

        u2 = np.array([[0, 0, 0, 0],
                      [0, x**3, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])

        u3 = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                       [0, 0, x**3, 0],
                       [0, 0, 0, 0]])

        u4 = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, x**3]])

        u5 = np.array([[0, x**2, 0, 0],
                      [x**2, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])

        u6 = np.array([[0, 0, x**2, 0],
                      [0, 0, 0, 0],
                       [x**2, 0, 0, 0],
                       [0, 0, 0, 0]])

        u7 = np.array([[0, 0, 0, x],
                       [0, 0, 0, 0],
                       [0, 0, 0, x],
                       [x, 0, x, 0]])

        u8 = np.array([[0, 0, 0, 0],
                       [0, 0, 0, x**3],
                       [0, 0, 0, 0],
                       [0, x**3, 0, 0]])

        u9 = np.array([[0, 0, 0, 0],
                       [0, 0, x, 0],
                       [0, x, 0, 0],
                       [0, 0, 0, 0]])

        u10 = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, x**2],
                       [0, 0, x**2, 0]])

        basis = [u1, u2, u3, u4, u5, u6, u7, u8, u9, u10]
        return basis


class DistanceFeatureMap(FeatureMapBase):
    """
    Feature Map inspired bt "Covariance Model with General Linear Structure
    and Divergent Parameters
    X assumed to be a pxK matrix where p is the dimension of the response
    variable Y
    """
    def __call__(self, x):
        p, K = x.shape
        basis = []
        basis.append(np.eye(p))
        for k in range(K):
            # Compute weight matrix W
            i, j = np.ogrid[:p, :p]
            wk = np.exp(-(x[i, k] - x[j, k])**2)
            np.fill_diagonal(wk, 0)
            basis.append(wk)
        return np.array(basis)


class FeatureMapFactory:
    # Define a feature map factory to produce concrete feature maps
    @staticmethod
    def create_feature_maps(map_type):
        if map_type == "example_feature_map_1":
            return ExampleFeatureMap1()
        elif map_type == "example_feature_map_2":
            return ExampleFeatureMap2()
        elif map_type == "example_feature_map_3":
            return ExampleFeatureMap3()
        elif map_type == "example_feature_map_4":
            return ExampleFeatureMap4()
        elif map_type == "example_feature_map_5":
            return ExampleFeatureMap5()
        elif map_type == "example_feature_map_6":
            return ExampleFeatureMap6()
        elif map_type == "distance_feature_map":
            return DistanceFeatureMap()
        else:
            raise ValueError(f"Unknown feature map type: {map_type}")

