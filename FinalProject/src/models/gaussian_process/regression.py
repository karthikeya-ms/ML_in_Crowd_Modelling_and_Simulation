"""
This module implements regression using gaussian processes.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.models.ml_model import MlModel
from src.models.gaussian_process.kernels import Kernel

class TrainingData(NamedTuple):
    """Container for gaussian process training data.

    Attributes:
    -----------
    cov_matrix: NDArray[np.float64]
        Also called kernel matrix.
    cov_matrix_inv: NDArray[np.float64]
        Inverted version for efficiency.
    X: NDArray[np.float64]
        The training datapoints.
    y: NDArray[np.float64]
        The training labels.
    input_dim: int 
        The number of input dimentions.
    output_dim: int
        The number of output dimentions.
    """
    cov_matrix: NDArray[np.float64]
    cov_matrix_inv: NDArray[np.float64]
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    input_dim: int
    output_dim: int



class GaussianProcessRegressor(MlModel):
    """An implementation of a Gaussian Process for regression.
    
    Attributes:
    -----------
    kernel: Kernel
        The kernel used by the GP for predictions.
    noise_variance: float
        The variance associated with the noise of the observations.
    training_data: TrainingData
        A named tuple with all of the training information of the GP.
    """

    def __init__(self, kernel: Kernel, noise_variance: np.float64 =0) -> None:
        self.kernel = kernel
        self.noise_variance = noise_variance
        
        self.training_data = None

    @property
    def log_marginal_likelihood(self) -> np.float64:
        """Calculates the log marginal likelihood.

        Raises:
            ValueError: If there is no training data.

        Returns:
            np.float64: The value calculated.
        """
        if not self.is_trained:
            raise ValueError('Gaussian process hasn\'t been trained.')

        return -0.5*self.training_data.y.T@self.training_data.cov_matrix_inv@self.training_data.y \
               -0.5*np.log(np.linalg.det(self.training_data.cov_matrix)) \
               -(self.training_data.y.shape[0]/2)*np.log(2*np.pi)

    def _kernel_tensor(self, X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculates the kernel matrix(es) between all points in X1 and X2.

        Args:
            X1 (NDArray[np.float64]): The first group of points.
            X2 (NDArray[np.float64]): The second group of points.

        Returns:
            NDArray[np.float64]: The resulting kernel tensor (in case of multiple matrixes).
        """
        kernel_tensor = np.zeros((X1.shape[0], X2.shape[0]))
        for i, x1_i in enumerate(X1):
            for j, x2_j in enumerate(X2):
                kernel_tensor[i,j] = self.kernel(x1_i, x2_j)
        return kernel_tensor

    def is_trained(self) -> bool:
        """Informs wether the gaussian process is trained.

        Returns:
            bool: True if the gaussian process is trained. False otherwise.
        """
        return self.training_data is not None

    def _mean(self, kernel_vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean prediction for a kernel vector.

        Args:
            kernel_vector (NDArray[np.float64]): The kernel vector for which the mean will be calculated.

        Returns:
            NDArray[np.float64]: The calculated mean.
        """
        return kernel_vector @ self.training_data.cov_matrix_inv @ self.training_data.y

    def _cov(
        self,
        kernel_vector: NDArray[np.float64],
        target_kernel_vector: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        """Covariance predicted for a kernel vector and target kernel vector.

        Args:
            kernel_vector (NDArray[np.float64]): The kernel vector for which the covariance 
                                                 matrix will be calculated.
            target_kernel_vector (NDArray[np.float64]): The vector of kernels between all targets.

        Returns:
            NDArray[np.float64]: The calculated covariance matrix.
        """
        return target_kernel_vector \
             - kernel_vector @ self.training_data.cov_matrix_inv @ kernel_vector.T

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """The MlModel method for training the model.
        
        Receives training data and fits the gaussian process to it.

        Args:
            X (NDArray[np.float64]): The training datapoints.
            y (NDArray[np.float64]): The training labels.
        """
        input_dim = 1 if len(X.shape) == 1 else X.shape[1]
        output_dim = 1 if len(y.shape) == 1 else y.shape[1]

        cov_matrix = self._kernel_tensor(X, X) + self.noise_variance * np.identity(X.shape[0])
        cov_matrix_inv = np.linalg.inv(cov_matrix)
        self.training_data = TrainingData(cov_matrix, cov_matrix_inv, X, y, input_dim, output_dim)

    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """The MlModel method for making a prediction on test data.

        Args:
            X (NDArray[np.float64]): The test data for which to predict.

        Returns:
            NDArray[np.float64]: The resulting prediction (mean) from the gaussian process.
        """
        kernel_vector = self._kernel_tensor(X, self.training_data.X)
        mean = self._mean(kernel_vector)

        return mean

    def get_distribution(self, X: NDArray[np.float64]) -> (NDArray[np.float64], NDArray[np.float64]):
        """A gaussian process method for prediction.

        Takes advantage of the gaussian process to return not only a prediction,
        but the confidence in the form of a gaussian distribution for the prediction.

        Args:
            X (NDArray[np.float64]): The test data fro which to predict.

        Returns:
            - means: The array with all of the means for predictions.
            - covs: The array with all of the covariance matrixes for predictions.
        """
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
