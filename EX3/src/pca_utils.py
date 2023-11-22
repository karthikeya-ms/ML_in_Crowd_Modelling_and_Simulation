from skimage.transform import rescale as skimage_rescale_func
import numpy as np
from pca import PCA


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
