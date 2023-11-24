import numpy as np
from scipy.linalg import sqrtm


def ambient_kernel(dataset: np.ndarray, diameter_percent: float = 0.05) -> np.ndarray:

    n_points = dataset.shape[0]
    distance_matrix = np.zeros((n_points, n_points), dtype=float)

    for i, point_i in np.ndenumerate(dataset):
        for j, point_j in np.ndenumerate(dataset):
            distance_matrix[i, j] = np.linalg.norm(point_i - point_j)

    epsilon = diameter_percent * np.matrix(distance_matrix).max()

    return np.exp(-np.power(distance_matrix, 2)/epsilon)


def normalize_kernel(ambient_kernel_: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Return normalized kernel T and the diagonal matrix & Q^{-1/2}
    """

    def sum_rows_diagonal(matrix: np.ndarray) -> np.ndarray:
        return np.diag(np.sum(matrix, axis=1))

    def diagonal_inverse_sandwich(diagonal: np.ndarray, kernel: np.ndarray, operator) -> np.ndarray:
        invert_diagonal = operator(diagonal)
        return invert_diagonal @ kernel @ invert_diagonal

    first_diagonal = sum_rows_diagonal(ambient_kernel_)
    kernel_matrix = diagonal_inverse_sandwich(first_diagonal, ambient_kernel_, np.invert)
    second_diagonal = sum_rows_diagonal(kernel_matrix)
    return (diagonal_inverse_sandwich(second_diagonal, kernel_matrix, lambda m: sqrtm(np.invert(m))),
            sqrtm(np.invert(second_diagonal)))


def largest_l_eigenvalues(matrix: np.ndarray, num: int = -1) -> (np.ndarray, np.ndarray):

    eigen_values, eigen_vectors = np.linalg.eig(matrix)

    # This assert should pass because the kernel matrix is real and symmetric
    # which means real eigenvalues and orthogonal eigenvectors
    assert np.all(np.isreal(eigen_values))
    assert np.all(np.isreal(eigen_vectors))

    # Sort by magnitude, desc. order
    sorted_indices = np.argsort(np.abs(eigen_values))[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]

    if num <= 0:
        return eigen_values, eigen_vectors
    elif num >= eigen_values.shape[0]:
        raise ValueError(f"num={num} is too large. There are only {eigen_values.shape[0]} eigenvalues")

    return eigen_values[:num + 1], eigen_vectors[:, :num + 1]


def diffusion_map(dataset: np.ndarray, diameter_percent: float = 0.05, num_eigenvalues: int = -1) -> (np.ndarray, np.ndarray):
    """
    Return lambda^2 and phi_l from the paper
    """

    kernel = ambient_kernel(dataset, diameter_percent)
    T, Q_inverse_sqrt = normalize_kernel(kernel)
    eigenvalues_T, eigenvectors_T = largest_l_eigenvalues(T, num_eigenvalues)

    # 9. Compute the eigenvalues of T hat^{1/epislon}
    # TODO: They are squared. Why?
    # These are the lambdas in the paper
    eigenvalues = np.power(eigenvalues_T, 1/diameter_percent)
    # These are the phi_l in the paper
    eigenvectors = Q_inverse_sqrt @ eigenvectors_T

    # Later, we may remove the first eigenvector
    # because it's constant if the data is connected for the supplied `diameter_percent`
    return eigenvalues, eigenvectors
