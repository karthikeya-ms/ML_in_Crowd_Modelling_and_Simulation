from skimage.transform import rescale as skimage_rescale_func
from numpy import ndarray


def rescale_img(greyscale_img: ndarray, new_width: int, new_height: int) -> ndarray:
    original_height, original_width = greyscale_img.shape[:2]

    scale_height = new_height / original_height
    scale_width = new_width / original_width

    greyscale_img_rescaled = skimage_rescale_func(greyscale_img, (scale_height, scale_width), mode='reflect')
    return greyscale_img_rescaled