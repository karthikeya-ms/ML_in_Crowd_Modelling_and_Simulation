"""
This module defines an abstract class for machine learning models.
"""
from abc import abstractmethod, ABC

import numpy as np
from numpy.typing import NDArray


class MlModel(ABC):
    """Abstract base class that defines a common api for machine learning models."""

    @abstractmethod
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Trains your model.

        Args:
            X (NDArray[np.float64]): The training dataset
            y (NDArray[np.float64]): The training labels
        """

    @abstractmethod
    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Makes prediction for new inputs using the model.

        Args:
            X (NDArray[np.float64]): The new inputs value.

        Returns:
            NDArray[np.float64]: The prediction of the model.
        """
