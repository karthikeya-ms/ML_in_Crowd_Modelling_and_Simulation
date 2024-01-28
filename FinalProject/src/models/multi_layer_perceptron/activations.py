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
        return 1 / (1 + np.exp(-Z))

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        return self(Z) * (1 - self(Z))


class Swish(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        s = Sigmoid()
        return Z * s(Z)

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        s = Sigmoid()
        return self(Z) + s(Z) * (1 - self(Z))


class Tanh(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tanh(Z)

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 - np.power(np.tanh(Z), 2)


class ReLU(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, Z)

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        return (Z > 0) * 1.0


class Softmax(Activation):
    def __init__(self):
        pass

    def __call__(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        sumAL = np.sum(Z, axis=1).reshape(-1, 1)
        return Z / sumAL

    def backwards(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("")
