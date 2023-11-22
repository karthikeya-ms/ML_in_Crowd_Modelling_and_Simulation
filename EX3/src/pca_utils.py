from skimage.transform import rescale as skimage_rescale_func
import numpy as np
from pca import PCA
import matplotlib.pyplot as plt


def rescale_img(greyscale_img: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    original_height, original_width = greyscale_img.shape[:2]

    scale_height = new_height / original_height
    scale_width = new_width / original_width

    greyscale_img_rescaled = skimage_rescale_func(greyscale_img, (scale_height, scale_width), mode='reflect')
    return greyscale_img_rescaled


def print_pca_info(pca: PCA, n: int = -1) -> None:
    pca.validate()

    if n <= 0:
        n = pca.S.shape[0]

    print(f'There are {pca.S.shape[0]} principal components. Ordered by magnitude:')
    print(f'First {n} singular values: {pca.S[:n]}')
    print(f'First {n} energies: {pca.energy[:n]}')
    print(f'Sum of largest {n} energies: {np.sum(pca.energy[:n])}')
    print(f'Sum of all energies should be 1: {np.sum(pca.energy)}')

def plot_reconstructed_image(pca_result, num_components, original_shape):
    reconstructed_image = pca_result.reverse_pca(r=num_components)
    plt.imshow(reconstructed_image.reshape(original_shape), cmap='gray')
    plt.title(f'Reconstruction with {num_components} Components')
    plt.axis('off')
    plt.show()


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