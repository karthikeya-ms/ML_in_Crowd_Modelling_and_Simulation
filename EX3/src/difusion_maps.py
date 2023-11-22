import numpy as np
from scipy.linalg import sqrtm

def ambient_kernel(dataset: np.array, diameter_percent: float =0.05):

    kernel = np.zeros((dataset.shape[0], dataset.shape[0]), dtype=float)

    for i, point_i in np.ndenumerate(dataset):
        for j, point_j in np.ndenumerate(dataset):
            kernel[i, j] = np.linalg.norm(point_i - point_j)

    epsilon = diameter_percent * np.matrix(kernel).max()

    return np.exp(-np.power(kernel, 2)/epsilon)

def normalize_kernel(ambient_kernel: np.array):
    
    def sum_rows_diagonal(matrix: np.array) -> np.array:
        kernel_side = ambient_kernel.shape[0]
        diagonal_sum_rows = np.zeros((kernel_side, kernel_side), dtype=float)
        sums = matrix.sum(axis=1)
        for i in range(kernel_side):
            diagonal_sum_rows[i,i] = sums[i]

        return diagonal_sum_rows

    def diagonal_inverse_sandwich(diagonal: np.array, kernel: np.array, operator) -> np.array:
        invert_diagonal = operator(diagonal)
        return invert_diagonal @ kernel @ invert_diagonal
    
    first_diagonal = sum_rows_diagonal(ambient_kernel)
    kernel_matrix = diagonal_inverse_sandwich(first_diagonal, ambient_kernel, np.invert)
    second_diagonal = sum_rows_diagonal(kernel_matrix)
    return diagonal_inverse_sandwich(second_diagonal, kernel_matrix, lambda m: sqrtm(np.invert(m)))

def eigen_calculator():
    eigen_values, eigen_vectors = np.linalg.eig()