from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.models.ml_model import MlModel
from src.models.gaussian_process.kernels import Kernel


class TrainingData(NamedTuple):
    cov_matrix: NDArray[np.float64]
    cov_matrix_inv: NDArray[np.float64]
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    input_dim: int
    output_dim: int


class GaussianProcess(MlModel):

    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.training_data = None

    def _kernel_tensor(self, X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
        kernel_tensor = np.zeros((X1.shape[0], X2.shape[0]))
        for i, x1_i in enumerate(X1):
            for j, x2_j in enumerate(X2):
                kernel_tensor[i,j] = self.kernel(x1_i, x2_j)
        return kernel_tensor

    def is_trained(self) -> bool:
        return self.training_data is not None
