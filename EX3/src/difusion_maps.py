import numpy as np
import scipy as s
from scipy.linalg import sqrtm
from scipy.spatial import KDTree
from scipy import sparse


def ambient_kernel(dataset: np.ndarray, diameter_percent: float = 0.05) -> sparse.csr_matrix:

    max_distance = 0
    for i in range(dataset.shape[0]):
        for j in range(0, i+1):
            new_distance = np.linalg.norm(dataset[i] - dataset[j])
            if new_distance > max_distance:
                max_distance = new_distance

    epsilon = diameter_percent * max_distance

    kd_tree = KDTree(dataset)
    sparse_distance_matrix = sparse.csr_matrix(kd_tree.sparse_distance_matrix(kd_tree, epsilon))

    sparse_distance_matrix.data = np.exp(-np.power(sparse_distance_matrix.data, 2) / epsilon)
    return sparse_distance_matrix


def normalize_kernel(ambient_kernel_: sparse.csr_matrix) -> (np.ndarray, np.ndarray):
    """
    Return normalized kernel T and the diagonal matrix & Q^{-1/2}
    """

    def sum_rows_diagonal(matrix: sparse.csr_array) -> sparse.csr_array:
        return sparse.csr_matrix(np.diag(
            np.array(matrix.sum(axis=1)).flatten()
        ))

    def kernel_sandwich(
        diagonal: sparse.csr_array,
        kernel: sparse.csr_array
    ) -> (
        sparse.csr_array, sparse.csr_matrix
    ):
        return diagonal @ kernel @ diagonal

    first_diagonal = s.sparse.linalg.inv(sum_rows_diagonal(ambient_kernel_))
    kernel_matrix = kernel_sandwich(first_diagonal, ambient_kernel_)
    second_diagonal = (s.sparse.linalg.inv(sum_rows_diagonal(kernel_matrix))).power(0.5)
    return (
        kernel_sandwich(second_diagonal, kernel_matrix),
        second_diagonal
    )


def largest_l_eigenvalues(matrix: sparse.csr_matrix, num: int = 0) -> (np.ndarray, np.ndarray):

    if num > matrix.shape[0] - 2:
        raise ValueError(
            f"num={num} is too large. Can only claculate up to and including N-2 eigenvalues"
        )

    # scipy's eigs cannot calculate more than N-2 eigenvalues.
    # This method was chosen because its efficient and the last eignevalues will allways be irrelevant
    eigen_values, eigen_vectors = sparse.linalg.eigs(
        matrix,
        k=num if num > 0 else matrix.shape[0]-2, which='LM'
    )

    # This assert should pass because the kernel matrix is real and symmetric
    # which means real eigenvalues and orthogonal eigenvectors
    assert np.all(np.isreal(eigen_values))
    assert np.all(np.isreal(eigen_vectors))

    # Sort by magnitude, desc. order
    sorted_indices = np.argsort(np.abs(eigen_values))[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]

    return eigen_values, eigen_vectors


def diffusion_map(
    dataset: np.ndarray,
    diameter_percent: float = 0.05,
    num_eigenvalues: int = -1
    ) -> (
        sparse.csr_matrix, sparse.csr_matrix
    ):
    """
    Return lambda^2 and phi_l from the paper
    """

    kernel = ambient_kernel(dataset, diameter_percent)
    T, Q_inverse_sqrt = normalize_kernel(kernel)
    eigenvalues_T, eigenvectors_T = largest_l_eigenvalues(T, num_eigenvalues)

    # 9. Compute the eigenvalues of T hat^{1/epislon}
    # TODO: They are squared. Why?
    # These are the lambdas in the paper
    eigenvalues = np.power(eigenvalues_T, 1/diameter_percent).real
    # These are the phi_l in the paper
    eigenvectors = ((Q_inverse_sqrt @ eigenvectors_T).real)

    # Later, we may remove the first eigenvector
    # because it's constant if the data is connected for the supplied `diameter_percent`
    return eigenvalues, eigenvectors

class DiffusionMap:

    def __init__(
        self,
        data: np.ndarray,
        diameter_percent: float =0.05,
        num_eigenvalues: int =-1
        ) -> None:
        self.eigen_values, self.eigen_vectors = \
            diffusion_map(data, diameter_percent=diameter_percent, num_eigenvalues=num_eigenvalues)
        
        
        self.length = num_eigenvalues
        self.normalize_eigen_vectors()
        
        
        self.points = np.array([self.eigen_vectors[:,i] * self.eigen_values[i] for i in range(num_eigenvalues)]).T
        

    def __len__(self) -> int:
        return self.length

    def normalize_eigen_vectors(self) -> None:
        for i in range(len(self)):
            for element in self.eigen_vectors[:,i]:
                if element == 0:
                    continue

                elif element < 0:
                    self.eigen_vectors[:,i] = self.eigen_vectors[:,i] * -1

                break
        

    def eigen_value(self, i) -> float:
        return self.eigen_values[i]

    def eigen_vector(self, i) -> np.ndarray:
        return self.eigen_vectors[:,i]

    def eigen_pair(self, i) -> (float, np.ndarray):
        return self.eigen_value(i), self.eigen_vector(i)

    def point(self, i) -> float:
        return np.multiply(self.eigen_vectors[i])

    def __iter__(self):
        return zip(self.eigen_values, self.eigen_vectors).__iter__()

    def __call__(self, x: np.array) -> np.array:
        assert x.shape == self.eigen_vectors[0].shape

        x_diff_space = np.zeros(len(self),)
        for i, (eigen_value, eigen_vector) in enumerate(self):
            x_diff_space[i] = eigen_value * eigen_vector

        return x_diff_space