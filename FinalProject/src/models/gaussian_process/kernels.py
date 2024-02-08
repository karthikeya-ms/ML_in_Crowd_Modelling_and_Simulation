"""
This module implements multiple kernels for use in gaussian processes.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class Kernel(ABC):
    """Generic abstract base class for kernels."""

    @abstractmethod
    def __call__(self, x1: NDArray[float64], x2: NDArray[float64]) -> float64:
        """Implementation of the call to the distance kernel function.

        Args:
            x1 (NDArray[float64]): The first data point.
            x2 (NDArray[float64]): The second data point.

        Returns:
            float64: The kernel distance value calculated.
        """


class RadialBasisFunction(Kernel):
    """Implementation of the radial basis function kernel.

    Attributes:
    -----------
    variance: float
        The variance hyperparameter of the rbf.
    length_sq: float
        The square of the length hyperparameter. Its saved squared for efficiency.
    """

    def __init__(self, variance: float64 =1, length: float64 =1) -> None:
        self.variance = variance
        self.length_sq = length ** 2

    def __call__(self, x1: NDArray[float64], x2: NDArray[float64]) -> float64:
        """Implementation of the call to the distance kernel function.

        Args:
            x1 (NDArray[float64]): The first data point.
            x2 (NDArray[float64]): The second data point.

        Returns:
            float64: The kernel distance value calculated.
        """
        return np.exp(-np.sum(np.square(x1-x2)) / 2*self.length_sq) * self.variance


class Periodic(Kernel):
    """Implementation of the periodic kernel.

    Attributes:
    -----------
    variance: float
        The variance hyperparameter of the rbf.
    length_sq: float
        The square of the length hyperparameter. Its saved squared for efficiency.
    period: float
        The length of the period of the kernel.
    """

    def __init__(self, variance: float64 =1, length: float64 =1, period: float64 =1) -> None:
        self.variance = variance
        self.length_sq = length**2
        self.period = period

    def __call__(self, x1: NDArray[float64], x2: NDArray[float64]) -> float64:
        """Implementation of the call to the distance kernel function.

        Args:
            x1 (NDArray[float64]): The first data point.
            x2 (NDArray[float64]): The second data point.

        Returns:
            float64: The kernel distance value calculated.
        """
        return self.variance * np.exp(
            - (2*np.square(np.sin(np.pi* np.linalg.norm(x1-x2)/self.period)))
            / self.length_sq
            )


class ComposeMultiply(Kernel):
    """Implementation of a composed by multiplication kernel.
    
    Multiplies the result of all of the provided kernels.

    Attributes:
    -----------
    kernels: list[Kernel]
        The list of kernels to multiply.
    """

    def __init__(self, kernels: list[Kernel]) -> None:
        self.kernels = kernels

    def __call__(self, x1: NDArray[float64], x2: NDArray[float64]) -> float64:
        """Implementation of the call to the distance kernel function.

        Args:
            x1 (NDArray[float64]): The first data point.
            x2 (NDArray[float64]): The second data point.

        Returns:
            float64: The kernel distance value calculated.
        """
        acumulator = 1
        for k in self.kernels:
            acumulator *= k(x1,x2)

        return acumulator
