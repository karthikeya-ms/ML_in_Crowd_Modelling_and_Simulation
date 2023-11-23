# This module tells Python to ignore annotations; they are only for the reader
# Required to annotate the return type of the static method `pca`
from __future__ import annotations

import numpy as np

# Type alias
ndarray = np.ndarray


class PCA:
    # U, S, Vh as obtained from singular value decomposition such that data_matrix == U @ diag(S) @ Vh
    U: ndarray
    S: ndarray
    Vh: ndarray

    # The average data point, typically the average row. If the data was transposed before performing PCA,
    # then the average data point is the average column of the data mabtrix
    mean: ndarray

    # Whether the data was transposed before performing PCA
    # This flag is set by the caller of the `pca` static method
    # The flag is used in `reverse_pca` to determine whether or to transpose the reconstructed data
    _transpose: bool = False

    @property
    def energy(self):
        return self.S ** 2 / np.sum(self.S ** 2)

    def __init__(self, u, s, vh: ndarray, mean: ndarray, transpose: bool = False):
        self.U = u
        self.S = s
        self.Vh = vh
        self.mean = mean
        self._transpose = transpose
        # User can input anything; make sure that the dimensions are valid
        self.validate()

    def __iter__(self):
        yield self.U
        yield self.S
        yield self.Vh
        yield self.mean

    def validate(self) -> None:
        """
        Paranoid checks to make sure that the dimensions are valid
        """
        num_singular_values = self.S.shape[0]
        expected_num_singular_values = min(self.U.shape[0], self.Vh.shape[0])
        # U and Vh are 2D matrices; S is the 1D vector of singular values (including multiplicities)
        assert len(self.U.shape) == 2 and len(self.S.shape) == 1 and len(self.Vh.shape) == 2
        # The matrix product U @ diag(s) @ Vh must be defined, hence check the dimensions
        assert self.U.shape[0] == self.U.shape[1] and self.Vh.shape[0] == self.Vh.shape[1]
        assert num_singular_values == expected_num_singular_values
        # If we did everything correctly, the sum of all energies should be 1. Check that it is close to 1
        assert np.abs(sum(self.energy) - 1) < 1e-10
        # Check that U and Vh are orthogonal/unitary
        assert np.allclose(self.U @ self.U.T, np.eye(self.U.shape[0]))
        assert np.allclose(self.Vh @ self.Vh.T, np.eye(self.Vh.shape[0]))
        # Check that the mean is a 1D vector
        assert len(self.mean.shape) == 1
        # Check that the mean has the same dimension as a data point
        assert self.mean.shape[0] == self.Vh.shape[0]

    def reverse_pca(self, r: int = -1, add_mean: bool = True) -> ndarray:
        """
        Reconstruct data matrix from the PCA format
        @param  r : int    number of principal components to use. If r <= 0, use all principal components
        @param  add_mean   A debug flag. If True, add the mean back to the reconstructed data.
                           Otherwise, return centered data
        """
        self.validate()

        u, s, vh, mean = self

        N, n = u.shape[0], vh.shape[1]

        # if n<0, it means we want to reconstruct the original data
        if r <= 0:
            r = s.shape[0]
        elif r > s.shape[0]:
            raise ValueError(f"r={r} is too large. There are only {s.shape[0]} principal components")

        # Get the singular value matrix
        sr = np.zeros((N, n))

        # Copy the first r singular values into Sr
        # s is sorted by singular value magnitude, so we can just take the first r principal components
        sr[:r, :r] = np.diag(s[:r])

        ret_val = u @ sr @ vh

        if add_mean:
            ret_val += mean

        if self._transpose:
            ret_val = ret_val.T

        return ret_val

    def __str__(self):
        return f"PCAResult(U={self.U},s={self.S},V={self.Vh})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def pca(data_matrix: ndarray, treat_columns_as_datapoints: bool = False) -> PCA:
        """
        Converts a data matrix to PCA format, i.e. svd on centered data.
        :param   data_matrix : array, shape (N, n). Assume n >= 2 and N >= 2
        :param   treat_columns_as_datapoints: Treat `data_matrix` as a (n, N) matrix instead of (N, n).
                 Transposes the data matrix to use columns as datapoints instead of rows.
                 Set this flag to true if you want to treat columns as datapoints.
                 The `reverse_pca` method respects this flag.
        :return  PCA instance with U, s, Vh such that data_matrix == U @ diag(s) @ Vh,
                 U: array (N, N), unitary
                 S: (n,) array of singular values. diag(s) is the (N,m) matrix from literature
                 Vh: array (n, n), unitary
        """

        # Treat columns as datapoints
        if treat_columns_as_datapoints:
            data_matrix = data_matrix.T

        # 2D nontrivial matrix, at least 2x2
        assert len(data_matrix.shape) == 2 and data_matrix.shape[1] >= 2 and data_matrix.shape[0] >= 2

        # Average data point, vector \overline{x} \in \mathbb{R}^n
        average_datapoint = np.mean(data_matrix, axis=0)
        # Center our matrix: X_{ij} - \overline{x}_j
        centered_data = data_matrix - average_datapoint

        # perform singular value decomposition on `centered_data`
        # s is a 1D vector; diag(s) is a (N, n) matrix
        # u and v are unitary: v @ vh == 1
        u, s, vh = np.linalg.svd(centered_data, full_matrices=True)

        to_ret = PCA(u, s, vh, average_datapoint, treat_columns_as_datapoints)
        return to_ret
