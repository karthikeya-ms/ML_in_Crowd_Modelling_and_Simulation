import numpy as np

def reverse_pca(U: np.ndarray, S: np.ndarray, V: np.ndarray, r=-1) -> np.ndarray:
    """
    Reconstruct data matrix from the PSA format
    :param U:
    :param S:
    :param V:
    :return:
    """

    # if n<0, it means we want to reconstruct the original data

    if r < 0:
        r = S.shape[0]
    elif r == 0:
        return np.zeros((0, 0))

    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = V[:r, :]

    return Ur @ Sr @ Vr.T


def pca(data_matrix: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Converts a data matrix to PSA format, i.e. svd on centered data, and adding energy values
    :param data_matrix: A 2D numpy array of shape (N, n)
    :return: U, S, V such that data_matrix == U @ S @ V^T,
             and E the energies of all singular values
    """
    # nonempty matrix
    assert (data_matrix.shape[0] > 0)

    # 2D matrix
    assert (len(data_matrix.shape) == 2)

    # Average data point, vector \overline{x} \in \mathbb{R}^n
    average_vector = np.mean(data_matrix, axis=0)

    # Center our matrix: X_{ij} - \overline{x}_j
    # Alternatively, centered_data = data_matrix - data_matrix.mean(axis=0) is also possible
    centered_data = data_matrix - average_vector

    # perform singular value decomposition on `centered_data`
    # S is our diagonal matrix of singular values
    # V is orthogonal: V @ V^T == I
    # centered_data == U @ S @ V^T
    U, S, V = np.linalg.svd(centered_data)

    # Get the energies
    # Remember, sum=trace because S is diagonal
    energies = S ** 2 / np.sum(S ** 2)
    energies = np.array([energies[i] for i in range(energies.shape[0])])

    # If we do it right, our energies sum up to 1
    assert (np.abs(sum(energies) - 1) < 1e-10)

    return U, S, V, energies
