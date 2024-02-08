from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.models.gaussian_process.kernels import Kernel
from src.models.gaussian_process import GaussianProcess, TrainingData


class GaussianProcessRegressor(GaussianProcess):

    def __init__(self, kernel: Kernel, noise_variance: np.float64 =0, prior_mean: NDArray[np.float64] =0) -> None:
        super().__init__(kernel)
        self.noise_variance = noise_variance
        self.prior_mean = prior_mean

    @property
    def log_marginal_likelihood(self) -> np.float64:
        if not self.is_trained:
            raise ValueError('Gaussian process hasn\'t been trained.')

        return -0.5*self.training_data.y.T@self.training_data.cov_matrix_inv@self.training_data.y \
               -0.5*np.log(np.linalg.det(self.training_data.cov_matrix)) \
               -(self.training_data.y.shape[0]/2)*np.log(2*np.pi)

    def _mean(self, kernel_vector: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.prior_mean + kernel_vector @ self.training_data.cov_matrix_inv @ self.training_data.y

    def _cov(
        self,
        kernel_vector: NDArray[np.float64],
        target_kernel_vector: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        return target_kernel_vector \
             - kernel_vector @ self.training_data.cov_matrix_inv @ kernel_vector.T

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        input_dim = 1 if len(X.shape) == 1 else X.shape[1]
        output_dim = 1 if len(y.shape) == 1 else y.shape[1]

        cov_matrix = self._kernel_tensor(X, X) + self.noise_variance * np.identity(X.shape[0])
        cov_matrix_inv = np.linalg.inv(cov_matrix)
        self.training_data = TrainingData(cov_matrix, cov_matrix_inv, X, y, input_dim, output_dim)

    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        kernel_vector = self._kernel_tensor(X, self.training_data.X)
        mean = self._mean(kernel_vector)

        return mean

    def get_distribution(self, X: NDArray[np.float64]) -> (NDArray[np.float64], NDArray[np.float64]):
        means = np.zeros((X.shape[0], self.training_data.output_dim))
        covs = np.zeros((X.shape[0], self.training_data.output_dim)
                        if self.training_data.output_dim == 1
                        else (X.shape[0], self.training_data.input_dim, self.training_data.output_dim))

        for i, point in enumerate(X):
            point = np.array([point])
            kernel_vector = self._kernel_tensor(point, self.training_data.X)
            target_kernel_vector = self._kernel_tensor(point, point)

            means[i] = self._mean(kernel_vector)
            covs[i] = self._cov(kernel_vector, target_kernel_vector)

        return means, covs
