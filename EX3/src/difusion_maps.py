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


def normalize_kernel(ambient_kernel_: np.ndarray) -> np.ndarray:
    
    def sum_rows_diagonal(matrix: np.ndarray) -> np.ndarray:
        return np.diag(np.sum(matrix, axis=1))

    def diagonal_inverse_sandwich(diagonal: np.ndarray, kernel: np.ndarray, operator) -> np.ndarray:
        invert_diagonal = operator(diagonal)
        return invert_diagonal @ kernel @ invert_diagonal

    first_diagonal = sum_rows_diagonal(ambient_kernel_)
    kernel_matrix = diagonal_inverse_sandwich(first_diagonal, ambient_kernel_, np.invert)
    second_diagonal = sum_rows_diagonal(kernel_matrix)
    return diagonal_inverse_sandwich(second_diagonal, kernel_matrix, lambda m: sqrtm(np.invert(m)))


def eigen_calculator():
    eigen_values, eigen_vectors = np.linalg.eig()
