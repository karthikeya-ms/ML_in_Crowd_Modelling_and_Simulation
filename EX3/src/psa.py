import numpy as np


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
    centered_data = data_matrix - average_vector

    # perform singular value decomposition on `centered_data`
    # S is our diagonal matrix of singular values
    # V is orthogonal: V @ V^T == I
    # centered_data == U @ S @ V^T
    U, S, V = np.linalg.svd(centered_data)

    # Get the energies
    # Remember, sum=trace becase S is diagonal
    energies = S ** 2 / np.sum(S ** 2)
    energies = np.array([energies[i] for i in range(energies.shape[0])])

    assert (len(energies.shape) == 1)

    # If we do it right, our energies sum up to 1
    assert (np.abs(sum(energies) - 1) < 1e-10)

    return U, S, V, energies
