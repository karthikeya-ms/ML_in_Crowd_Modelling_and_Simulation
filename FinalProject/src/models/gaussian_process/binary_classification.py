import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor
from typing import Callable, NamedTuple

from src.models.gaussian_process import GaussianProcess, TrainingData
from src.models.gaussian_process.kernels import Kernel

# Assuming MlModel and Kernel are defined elsewhere
class GaussianProcessClassifier(GaussianProcess):

    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray], float]) -> None:
        super().__init__(kernel)

    def _logistic(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _fit_laplace_approximation(self, X: np.ndarray, y: np.ndarray):
        # Initialize the latent function values
        f_init = np.zeros_like(y)

        # Objective function for the Laplace approximation
        def obj_func(f):
            pi = self._logistic(f)
            return -np.sum(y * np.log(pi) + (1 - y) * np.log(1 - pi))  # Negative log likelihood

        # Gradient of the objective function
        def grad_func(f):
            pi = self._logistic(f)
            return -(y - pi)

        # Use a numerical optimizer to find the mode of the posterior
        f_opt = minimize(obj_func, f_init, method='L-BFGS-B', jac=grad_func).x

        # Compute the Hessian (second derivatives matrix) at the mode for the Laplace approximation
        pi_opt = self._logistic(f_opt)
        W = np.diag(pi_opt * (1 - pi_opt))
        Hessian = self._kernel_tensor(X, X) + np.linalg.inv(W)

        self.training_data = TrainingData(np.linalg.inv(Hessian), X, f_opt)  # Store the inverse Hessian and other training data

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._fit_laplace_approximation(X, y)

    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        K_star = self._kernel_tensor(X, self.training_data.X)
        f_star_mean = K_star @ self.training_data.cov_matrix_inv @ self.training_data.y

        # Convert latent function to probability using logistic function
        proba = self._logistic(f_star_mean)
        return proba
