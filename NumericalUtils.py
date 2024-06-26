"""

This is a python module to provide Numerical utilities to support the Entropic
Covariance Model. This includes various optimization routine, numerical
differentiations etc

Author: Dongli Chen, Department of Statistics, University of Toronto, dongli.chen@mail.utoronto.ca
        William Groff, Department of Statistics, University of Toronto, william.groff@mail.utoronto.ca
        Piotr Zwiernik, Department of Statistics, University of Toronto, piotr.zwiernik@utoronto.ca
"""
from abc import ABC, abstractmethod
import numpy as np
import multiprocessing as mp


class Penalty(ABC):
    """
    Interface for different type of penalty functions like l_2(Ridge),
    l_2(LASSO) etc
    """

    @abstractmethod
    def gradient(self, *args, **kwargs):
        pass


class L2_Penalty(Penalty):

    def gradient(self, alpha, center_pos):
        return alpha - center_pos


class Optimizer(ABC):
    """
    Define an abstract base class as an interface for optimizer.
    This allows flexible extensions on the family of optimization approaches
    in the future since one can implement a new concrete child class
    for a new optimization approach
    """

    def __init__(self):
        self.optimization_config = None

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass


class GradientDescent(Optimizer):
    """
    Performs gradient descent optimization.

    Fields:
    - gradient_function: Function to compute the gradient of the function to be minimized.
    - initial_guess: Initial starting point for the algorithm.
    - learning_rate: Learning rate (step size) for each iteration.
    - n_iter: Number of iterations to perform.
    - tolerance: Tolerance for stopping criteria.

    """

    def __init__(self, optimization_config):
        super().__init__()
        self.gradient_function = optimization_config['gradient']
        self.initial_guess = optimization_config['initial_guess']
        self.learning_rate = optimization_config['learning_rate']
        self.n_iter = optimization_config['n_iter']
        self.tolerance = optimization_config['tolerance']

    def optimize(self):
        """
        :return: vector of positions for each iteration
        """

        x = self.initial_guess
        positions = [x]

        for _ in range(self.n_iter):
            grad = self.gradient_function(x)
            x_new = x - self.learning_rate * grad
            positions.append(x_new)

            # Stop if the change is smaller than the tolerance
            if np.all(np.abs(x_new - x) <= self.tolerance):
                break
            x = x_new

        return np.array(positions)

    def set_gradient_function(self, gradient_function):
        self.gradient_function = gradient_function

    def get_gradient_function(self):
        return self.gradient_function


class SGD(Optimizer):
    """
    Performs stochastic gradient descent optimization.

    Fields:
    - gradient_function: Function to compute the gradient of the function to be minimized.
    - initial_guess: Initial starting point for the algorithm.
    - learning_rate: Learning rate (step size) for each iteration.
    - n_iter: Number of iterations to perform.
    - tolerance: Tolerance for stopping criteria.

    """

    def __init__(self, optimization_config):
        super().__init__()
        self.gradient_function = optimization_config['gradient']
        self.initial_guess = optimization_config['initial_guess']
        self.learning_rate = optimization_config['learning_rate']
        self.n_iter = optimization_config['n_iter']
        self.tolerance = optimization_config['tolerance']
        self.batch_size = optimization_config['batch_size']
        self.random_sample = optimization_config['random']
        self.num_samples = optimization_config['num_samples']

    def optimize(self):
        """
        :return: vector of positions for each iteration
        """

        x = self.initial_guess
        positions = [x]
        batch_counter = 0
        for it in range(self.n_iter):
            print(str(it) + '/' + str(self.n_iter))
            if self.random_sample:
                # Randomly samples batches of specified size
                # Slow for large sample size
                batch_indices = np.random.choice(self.num_samples,
                                                 size=self.batch_size,
                                                 replace=False)
            else:
                # Cycles through samples in order
                batch_indices = list(range(batch_counter*self.batch_size,
                                           (batch_counter + 1)*self.batch_size))

                batch_counter += 1
                if (batch_counter + 1)*self.batch_size > self.num_samples:
                    batch_counter = 0
            grad = self.gradient_function(x, batch_indices)
            x_new = x - self.learning_rate * grad
            positions.append(x_new)

            # Stop if the change is smaller than the tolerance
            if np.all(np.abs(x_new - x) <= self.tolerance):
                break
            x = x_new

        return np.array(positions)

    def set_gradient_function(self, gradient_function):
        self.gradient_function = gradient_function

    def get_gradient_function(self):
        return self.gradient_function


class ADMM(Optimizer):
    def __init__(self):
        super().__init__()
        self.single_thread_optimizer = None
        self.penalty_function = None
        self.center_pos = None
        self.ancillary_pos = None

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    """
    @abstractmethod
    def single_thread_optimize(self, *args, **kwargs):
        pass
        """


class ADMM_GD(ADMM):
    def __init__(self, optimization_config):
        super().__init__()
        self.single_thread_optimizer = GradientDescent(
            optimization_config["single_thread_optimizer_config"])
        self.penalty_function = PenaltyFactory.create_penalty_function(
            optimization_config["penalty_type"])
        self.penalty_rate = optimization_config["penalty_rate"]
        self.center_pos = np.zeros(optimization_config["alpha_dim"])
        self.ancillary_pos_list = np.zeros(
            (optimization_config["num_of_samples"],
             optimization_config["alpha_dim"]))
        self.alpha_list = np.zeros(
            (optimization_config["num_of_samples"],
             optimization_config["alpha_dim"]))
        self.max_iter = optimization_config["max_iter"]
        self.tolerance = optimization_config["tolerance"]
        self.loss_function_gradient = optimization_config[
            "loss_function_gradient"]
        self.num_of_samples = optimization_config["num_of_samples"]

    def optimize(self):
        print("Iterations:")
        for it in range(self.max_iter):
            print(str(it + 1) + "/" + str(self.max_iter))
            # batch updates of alphas
            self._batch_alpha_update()
            # update for the center position
            cumulative_center_pos = (self.alpha_list +
                                     (self.ancillary_pos_list / self.penalty_rate))
            new_center_pos = np.mean(cumulative_center_pos, axis=0)
            if np.all(np.abs(new_center_pos - self.center_pos)
                      <= self.tolerance):
                break
            self.center_pos = new_center_pos
            # batch updates of ancillary positions
            self._batch_ancillary_pos_update()
        return self.center_pos

    """
        Private helper methods for optimize
    """
    def _batch_alpha_update(self):
        process = []

        manager = mp.Manager()
        shared_alpha = manager.list([alpha.copy() for alpha in self.alpha_list])

        lock = mp.Lock()

        for i in range(self.num_of_samples):
            p = mp.Process(target=self._update_alpha_single_thread,
                           args=(i, shared_alpha, lock))
            process.append(p)
            p.start()
        for p in process:
            p.join()

        self.alpha_list = np.array(shared_alpha)

    def _batch_ancillary_pos_update(self):
        process = []

        manager = mp.Manager()
        shared_ancillary_pos = manager.list([ancillary_pos.copy() for ancillary_pos
                                             in self.ancillary_pos_list])

        lock = mp.Lock()

        for i in range(self.num_of_samples):
            p = mp.Process(target=self._update_ancillary_pos_single_thread,
                           args=(i, shared_ancillary_pos, lock))
            process.append(p)
            p.start()
        for p in process:
            p.join()

        self.ancillary_pos_lis = np.array(shared_ancillary_pos)

    def _update_alpha_single_thread(self, i, shared_alpha, lock):
        # Update the gradient function based on the current center_pos z^{k} and
        # current ancillary_pos w^{k}
        center_pos = self.center_pos
        ancillary_pos = self.ancillary_pos_list[i]

        def gradient_function(x):
            return (self.loss_function_gradient(x) +
                    self.penalty_rate * self.penalty_function.gradient(x,
                                                                       center_pos) +
                    ancillary_pos)

        self.single_thread_optimizer.set_gradient_function(gradient_function)
        # Only take final estimate of updated alpha
        updated_alpha = self.single_thread_optimizer.optimize()
        with lock:
            shared_alpha[i] = updated_alpha[-1]

    def _update_ancillary_pos_single_thread(self, i, shared_ancillary_pos, lock):
        updated_alpha = self.alpha_list[i]
        updated_center_post = self.center_pos
        curr_ancillary_post = self.ancillary_pos_list[i]
        updated_ancillary_pos = (curr_ancillary_post +
                                 self.penalty_rate *
                                 (updated_alpha - updated_center_post))

        with lock:
            shared_ancillary_pos[i] = updated_ancillary_pos


class ADMM_GD_NoPen(ADMM):
    def __init__(self, optimization_config):
        super().__init__()
        self.single_thread_optimizer = GradientDescent(
            optimization_config["single_thread_optimizer_config"])
        self.center_pos = np.zeros(optimization_config["alpha_dim"])
        self.ancillary_pos_list = np.zeros(
            (optimization_config["num_of_samples"],
             optimization_config["alpha_dim"]))
        self.alpha_list = np.zeros(
            (optimization_config["num_of_samples"],
             optimization_config["alpha_dim"]))
        self.max_iter = optimization_config["max_iter"]
        self.tolerance = optimization_config["tolerance"]
        self.loss_function_gradient = optimization_config[
            "loss_function_gradient"]
        self.num_of_samples = optimization_config["num_of_samples"]

    def optimize(self):
        print("Iterations:")
        for it in range(self.max_iter):
            print(str(it + 1) + "/" + str(self.max_iter))
            # batch updates of alphas
            self._batch_alpha_update()
            # update for the center position
            cumulative_center_pos = self.alpha_list
            new_center_pos = np.mean(cumulative_center_pos, axis=0)
            if np.all(np.abs(new_center_pos - self.center_pos)
                      <= self.tolerance):
                break
            self.center_pos = new_center_pos
            # batch updates of ancillary positions
        return self.center_pos

    """
        Private helper methods for optimize
    """
    def _batch_alpha_update(self):
        process = []

        manager = mp.Manager()
        shared_alpha = manager.list([alpha.copy() for alpha in self.alpha_list])

        lock = mp.Lock()

        for i in range(self.num_of_samples):
            p = mp.Process(target=self._update_alpha_single_thread,
                           args=(i, shared_alpha, lock))
            process.append(p)
            p.start()
        for p in process:
            p.join()

        self.alpha_list = np.array(shared_alpha)

    def _batch_ancillary_pos_update(self):
        process = []

        manager = mp.Manager()
        shared_ancillary_pos = manager.list([ancillary_pos.copy() for ancillary_pos
                                             in self.ancillary_pos_list])

        lock = mp.Lock()

        for i in range(self.num_of_samples):
            p = mp.Process(target=self._update_ancillary_pos_single_thread,
                           args=(i, shared_ancillary_pos, lock))
            process.append(p)
            p.start()
        for p in process:
            p.join()

        self.ancillary_pos_lis = np.array(shared_ancillary_pos)

    def _update_alpha_single_thread(self, i, shared_alpha, lock):
        # Update the gradient function based on the current center_pos z^{k} and
        # current ancillary_pos w^{k}
        center_pos = self.center_pos
        ancillary_pos = self.ancillary_pos_list[i]

        def gradient_function(x):
            return (self.loss_function_gradient(x))

        self.single_thread_optimizer.set_gradient_function(gradient_function)
        # Only take final estimate of updated alpha
        updated_alpha = self.single_thread_optimizer.optimize()
        with lock:
            shared_alpha[i] = updated_alpha[-1]


class OptimizerFactory:
    # Define an Optimizer factory to produce various optimizer
    @staticmethod
    def create_optimizer(optimizer_type, optimization_config, target_model):
        if optimizer_type == "GradientDescent":
            optimization_config["gradient"] = target_model.compute_gradient
            return GradientDescent(optimization_config)
        elif optimizer_type == "SGD":
            optimization_config["gradient"] = target_model.compute_batch_gradient
            return SGD(optimization_config)
        elif optimizer_type == "ADMM GD":
            optimization_config["loss_function_gradient"] = target_model.compute_gradient
            optimization_config["single_thread_optimizer_config"]["gradient"] = None
            return ADMM_GD(optimization_config)
        elif optimizer_type == "ADMM GD No Pen":
            optimization_config["loss_function_gradient"] = target_model.compute_gradient
            optimization_config["single_thread_optimizer_config"]["gradient"] = None
            return ADMM_GD_NoPen(optimization_config)
        else:
            raise ValueError(f"Unknown Optimizer type: {optimizer_type}")


class PenaltyFactory:
    @staticmethod
    def create_penalty_function(penalty_type):
        if penalty_type == "L2":
            return L2_Penalty()
        else:
            raise ValueError(f"Unknown Penalty Function type: {penalty_type}")
