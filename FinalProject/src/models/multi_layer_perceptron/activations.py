"""Contains all code related to activation functions."""

from abc import abstractmethod, ABC
import numpy as np
from numpy.typing import NDArray


class Activation(ABC):
    @abstractmethod
    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply a function on the activations.

        Args:
            Z (NDArray[np.float64]): The activation array.
            y (NDArray[np.float64]): The training labels.

        Returns:
            NDArray[np.float64]: The non-linear activations.
        """

    @abstractmethod
    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply a backward step on the activations.

        Args:
            Z (NDArray[np.float64]): The activation array.

        Returns:
            NDArray[np.float64]: the derivative w.r.t. the activations Z.
        """


class Sigmoid(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the sigmoid activation for the given input.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Output values after applying the sigmoid activation.
        """
        return 1 / (1 + np.exp(-Z))

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the derivative of the sigmoid activation.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Derivative of the sigmoid activation.
        """
        return self(Z) * (1 - self(Z))


class Swish(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the Swish activation for the given input.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Output values after applying the Swish activation.
        """
        s = Sigmoid()
        return Z * s(Z)

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the derivative of the Swish activation.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Derivative of the Swish activation.
        """
        s = Sigmoid()
        return self(Z) + s(Z) * (1 - self(Z))


class Tanh(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the hyperbolic tangent (tanh) activation for the given input.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Output values after applying the tanh activation.
        """
        return np.tanh(Z)

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the derivative of the tanh activation.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Derivative of the tanh activation.
        """
        return 1.0 - np.power(np.tanh(Z), 2)


class ReLU(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the Rectified Linear Unit (ReLU) activation for the given input.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Output values after applying the ReLU activation.
        """
        return np.maximum(0, Z)

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the derivative of the ReLU activation.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Derivative of the ReLU activation.
        """
        return (Z > 0) * 1.0


class Softmax(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the softmax activation for the given input.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Output values after applying the softmax activation.
        """
        Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        sumAL = np.sum(Z, axis=1).reshape(-1, 1)
        return Z / sumAL

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Placeholder for the derivative of the softmax activation. Not implemented 
        in favor of taking the derivative of the cross-entropy loss with Softmax in one step.

        Parameters:
        - Z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Derivative of the softmax activation.
        """
        raise NotImplementedError("Backward pass for Softmax not implemented.")

