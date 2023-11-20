import numpy as np


class PSAResult:
    U: np.ndarray
    S: np.ndarray
    Vh: np.ndarray

    @property
    def energy(self):
        return self.S ** 2 / np.sum(self.S ** 2)

    def __init__(self, U, s, Vh):
        self.U = U
        self.S = s
        self.Vh = Vh
        self.validate()

    def __iter__(self):
        yield self.U
        yield self.S
        yield self.Vh

    def validate(self):
        assert len(self.U.shape) == 2 and len(self.S.shape) == 1 and len(self.Vh.shape) == 2
        assert self.U.shape[0] == self.U.shape[1] and self.Vh.shape[0] == self.Vh.shape[1]
        assert self.S.shape[0] == self.Vh.shape[0]
        assert np.abs(sum(self.energy) - 1) < 1e-10

    def __str__(self):
        return f"PSAResult(U={self.U},s={self.S},V={self.Vh})"

    def __repr__(self):
        return self.__str__()


def reverse_pca(u: np.ndarray, s: np.ndarray, vh: np.ndarray, r=-1) -> np.ndarray:
    """
    Reconstruct data matrix from the PSA format
    :param u:  Orthogonal (N,N) matrix
    :param s:  Semi-diagonal (n,) matrix
    :param vh: Orthogonal (n,n) matrix
    :param r:  Number of singular values to use. If r<=0, use all singular values
    :return:   reconstructed array, (N,n)

    The parameters must have valid dimensions, i.e. the product U @ diag(s) @ Vh must be defined

    """

    # dimensions
    assert len(u.shape) == 2 and len(s.shape) == 1 and len(vh.shape) == 2
    # square matrices
    assert u.shape[0] == u.shape[1] and vh.shape[0] == vh.shape[1]
    assert s.shape[0] == vh.shape[0]

    N, n = u.shape[0], vh.shape[1]

    # if n<0, it means we want to reconstruct the original data
    if r <= 0:
        r = s.shape[0]

    # Get the singular value matrix
    Sr = np.zeros((N, n))
    # Copy the first r singular values into Sr
    # for i in range(s.shape[0]):
    #     if i >= r:
    #         break
    #     Sr[i, i] = s[i]

    Sr[:r, :r] = np.diag(s[:r])

    return u @ Sr @ vh


def pca(data_matrix: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Converts a data matrix to PSA format, i.e. svd on centered data.
    :param   data_matrix : array, shape (N, n). Assume n >= 2 and N >= 2
    :return: U, s, Vh such that data_matrix == U @ diag(s) @ Vh,
             and E the energies of all singular values
    :return U: array (N, N), unitary
    :return S: (n,) array of singular values. diag(s) is the (N,m) matrix from literature
    :return Vh: array (n, n), unitary
    :return E: (n,) array with the energies of all singular values

    """
    # 2D nontrivial matrix
    assert len(data_matrix.shape) == 2 and data_matrix.shape[1] >= 2 and data_matrix.shape[0] >= 2

    N, n = data_matrix.shape[0], data_matrix.shape[1]

    # Average data point, vector \overline{x} \in \mathbb{R}^n
    average_vector = np.mean(data_matrix, axis=0)
    # Center our matrix: X_{ij} - \overline{x}_j
    centered_data = data_matrix - average_vector

    # perform singular value decomposition on `centered_data`
    # s is a 1D vector; diag(s) is a (N, n) matrix
    # U and V are unitary: V @ V.H == I
    # centered_data == U @ diag(s) @ V.H
    U, s, Vh = np.linalg.svd(centered_data, full_matrices=True)

    # Get the energies
    energies = s ** 2 / np.sum(s ** 2)

    # TODO: Move these checks to unit tests
    # ----------------------- Checks -----------------------
    S = np.zeros((N, n))
    # Copy s into S
    S[:n, :n] = np.diag(s)
    reconstructed_data = U @ S @ Vh

    # Can't do that since Python optimizes and avoids returning the full matrix
    # assert S.shape == data_matrix.shape
    assert U.shape == (data_matrix.shape[0], data_matrix.shape[0])
    assert Vh.shape == (data_matrix.shape[1], data_matrix.shape[1])
    # Energies are 1D and singular value array are 1D
    assert len(s.shape) == 1
    assert len(energies.shape) == 1
    assert S.shape == data_matrix.shape
    # If we do it right, our energies sum up to 1
    assert (np.abs(sum(energies) - 1) < 1e-10)
    assert np.allclose(centered_data, reconstructed_data)
    # END ----------------------- Checks -----------------------

    return U, s, Vh, energies
