# This module tells Python to ignore annotations; they are only for the reader
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class PCA:
    U: np.ndarray
    S: np.ndarray
    Vh: np.ndarray
    _mean = None

    @property
    def mean(self) -> np.ndarray:
        if self._mean is None:
            # throw error
            raise ValueError("PCA not performed")
        return self._mean

    @property
    def energy(self):
        return self.S ** 2 / np.sum(self.S ** 2)

    def __init__(self, u, s, vh):
        self.U = u
        self.S = s
        self.Vh = vh
        self._mean = None
        # User can input anything; make sure that the dimensions are valid
        self.validate()

    def __iter__(self):
        yield self.U
        yield self.S
        yield self.Vh

    def validate(self):
        # U and Vh are 2D matrices; S is the 1D vector of singular values (including multiplicities)
        assert len(self.U.shape) == 2 and len(self.S.shape) == 1 and len(self.Vh.shape) == 2
        # The matrix product U @ diag(s) @ Vh must be defined, hence check the dimensions
        assert self.U.shape[0] == self.U.shape[1] and self.Vh.shape[0] == self.Vh.shape[1]
        assert self.S.shape[0] == self.Vh.shape[0]
        # If we did everything correctly, the energy should be 1. Check that it is close to 1
        assert np.abs(sum(self.energy) - 1) < 1e-10
        # Check that U and Vh are orthogonal/unitary
        assert np.allclose(self.U @ self.U.T, np.eye(self.U.shape[0]))
        assert np.allclose(self.Vh @ self.Vh.T, np.eye(self.Vh.shape[0]))
        if self._mean is not None:
            assert isinstance(self._mean, np.ndarray)
            assert len(self._mean.shape) == 1
            assert self._mean.shape[0] == self.Vh.shape[0]

    def reverse_pca(self, r: int = -1, add_mean: bool = True) -> np.ndarray:
        """
        Reconstruct data matrix from the PCA format
        @param  r : int    number of principal components to use. If r <= 0, use all principal components
        @param  add_mean   if True, add the mean to the reconstructed data. Otherwise, return centered data
        """
        self.validate()
        u, s, vh = self

        N, n = u.shape[0], vh.shape[1]

        # if n<0, it means we want to reconstruct the original data
        if r <= 0:
            r = s.shape[0]

        # Get the singular value matrix
        sr = np.zeros((N, n))

        # Copy the first r singular values into Sr
        # s is sorted by singular value magnitude, so we can just take the first r principal components
        sr[:r, :r] = np.diag(s[:r])

        ret_val = u @ sr @ vh

        if add_mean:
            ret_val += self._mean

        return ret_val

    def __str__(self):
        return f"PCAResult(U={self.U},s={self.S},V={self.Vh})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def pca(data_matrix: np.ndarray) -> PCA:
        """
        Converts a data matrix to PCA format, i.e. svd on centered data.
        :param   data_matrix : array, shape (N, n). Assume n >= 2 and N >= 2
        :return  PCAResult: U, s, Vh such that data_matrix == U @ diag(s) @ Vh,
                 U: array (N, N), unitary
                 S: (n,) array of singular values. diag(s) is the (N,m) matrix from literature
                 Vh: array (n, n), unitary
        """
        # 2D nontrivial matrix
        assert len(data_matrix.shape) == 2 and data_matrix.shape[1] >= 2 and data_matrix.shape[0] >= 2

        # Average data point, vector \overline{x} \in \mathbb{R}^n
        average_datapoint = np.mean(data_matrix, axis=0)
        # Center our matrix: X_{ij} - \overline{x}_j
        centered_data = data_matrix - average_datapoint

        # perform singular value decomposition on `centered_data`
        # s is a 1D vector; diag(s) is a (N, n) matrix
        # u and v are unitary: v @ vh == 1
        u, s, vh = np.linalg.svd(centered_data, full_matrices=True)

        to_ret = PCA(u, s, vh)
        to_ret._mean = average_datapoint
        return to_ret


def plot_data_with_pcs(data_centered, Vt) -> None:
    """
    Plot the data and the principal components

    :param data_centered: array, shape (N, n)
    """
    # Plot the Data
    plt.scatter(data_centered[:, 0], data_centered[:, 1])

    # Add Principal Components
    mean_data = data_centered.mean(axis=0)
    plt.quiver(mean_data[0], mean_data[1], Vt[0, 0], Vt[0, 1], scale=3, color='r')
    plt.quiver(mean_data[0], mean_data[1], Vt[1, 0], Vt[1, 1], scale=3, color='g')
    plt.title('PCA of Dataset')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.axis('equal')

    # Show plot with principal components
    plt.show()

