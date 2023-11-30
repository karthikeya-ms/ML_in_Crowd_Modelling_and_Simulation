from skimage.transform import rescale as skimage_rescale_func
import numpy as np
from pca import PCA
import matplotlib.pyplot as plt

# Documentation of module:
# This module contains utility functions for the PCA Python notebook


def print_pca_info(pca: PCA, n: int = -1) -> None:
    pca.validate()

    if n <= 0:
        n = pca.S.shape[0]

    print(f'There are {pca.S.shape[0]} principal components, ordered by magnitude')
    print(f'First {n} singular values: {pca.S[:n]}')
    print(f'First {n} energies: {pca.energy[:n]}')
    print(f'Sum of largest {n} energies: {np.sum(pca.energy[:n])}')


def plot_pedestrian_figure_variant1(data_matrix: np.ndarray,
                                    comment: str = "Original",
                                    original_matrix: np.ndarray | None = None) -> None:
    """
    Plot the trajectories of two pedestrians
    :param data_matrix: array, shape (N, n), where n >= 4
    :param comment: comment to add to the title
    :param original_matrix: Original matrix to plot the arrows of the first pedestrian (if reconstructed)
    """
    x1, y1, x2, y2 = extract_first_two_pedestrian_data(data_matrix)
    check_two_pedestrian_arrays(x1, y1, x2, y2)

    # Plotting the paths for the first two pedestrians
    plt.figure(figsize=(10, 6))
    plt.scatter(x1, y1, label='Pedestrian 1', marker='o')
    plt.scatter(x2, y2, label='Pedestrian 2', marker='o', color='orange')
    plot_trajectory_vectors(x1, y1, 75, 30, start_color='darkgreen', arrow_color='r', arrow_scale=10, zorder=3)
    # If we have the original matrix, plot the trajectory vectors for the original matrix in grey
    # Otherwise, assume that `data_matrix` is the original matrix
    if original_matrix is not None:
        assert original_matrix.shape == data_matrix.shape
        x1_original, y1_original, _, _ = extract_first_two_pedestrian_data(original_matrix)
        plot_trajectory_vectors(x1_original, y1_original, 45, 30, start_color='grey', arrow_color='grey', arrow_scale=15, zorder=1)
    plt.title(f'Two Pedestrians, {comment}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pedestrian_figure_variant2(data_matrix: np.ndarray,
                                    comment: str = "Original") -> None:
    """
    Plot the trajectories of two pedestrians
    :param data_matrix: array, shape (N, n), where n >= 4
    :param comment: comment to add to the title
    """
    x1, y1, x2, y2 = extract_first_two_pedestrian_data(data_matrix)
    check_two_pedestrian_arrays(x1, y1, x2, y2)

    # Plotting the paths for the first two pedestrians
    plt.figure(figsize=(10, 6))
    # Create a time array from 0 to the number of points for pedestrian 1
    t1 = list(range(len(x1)))
    # Create a scatter plot for pedestrian 1, with the color representing time
    plt.scatter(x1, y1, c=t1, label='Pedestrian 1', cmap='Blues')
    plt.colorbar(label='Timestep for Pedestrian 1')

    # Create a time array from 0 to the number of points for pedestrian 2
    t2 = list(range(len(x2)))
    # Create a scatter plot for pedestrian 2, with the color representing time
    plt.scatter(x2, y2, c=t2, label='Pedestrian 2', cmap='Greens')
    plt.colorbar(label='Timestep for Pedestrian 2')
    plt.title(f'Two Pedestrians, {comment}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def reconstruct_and_plot_trajectory(pca_result: PCA, num_components: int) -> None:
    reconstructed_data = pca_result.reverse_pca(r=num_components)
    comment = (f'{num_components} / {pca_result.S.shape[0]} components,'
               f'{pca_result.energy_until(num_components)}% energy')
    plot_pedestrian_figure_variant1(reconstructed_data, comment)
    plot_pedestrian_figure_variant1(reconstructed_data, comment + ', including original trajectory', pca_result.reverse_pca())
    plot_pedestrian_figure_variant2(reconstructed_data, comment)

def rescale_greyscale_img(greyscale_img: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    original_height, original_width = greyscale_img.shape[:2]

    scale_height = new_height / original_height
    scale_width = new_width / original_width

    return skimage_rescale_func(greyscale_img, (scale_height, scale_width), mode='reflect')


def plot_reconstructed_image(pca_result: PCA, num_components, original_shape):
    reconstructed_image = pca_result.reverse_pca(r=num_components)
    plt.imshow(reconstructed_image.reshape(original_shape), cmap='gray')
    # print title but with floating point round after 5 digits
    plt.title(f'{num_components} / {pca_result.S.shape[0]} Components, '
              f'Energy: {100*np.sum(pca_result.energy[:num_components]):.5f}%')
    plt.axis('off')
    plt.show()


def plot_data_with_pcs(data_centered: np.ndarray, Vt: np.ndarray) -> None:
    """
    Plot the data and the principal components

    :param data_centered: array, shape (N, n)
    :param Vt: array, shape (n, n). Orthogonal matrix, contains principal components
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


def extract_first_two_pedestrian_data(matrix: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Extract the first two pedestrian data from the matrix
    :param   matrix: array, shape (N, n), where n >= 4
    :return: x1, y1, x2, y2: array, shape (N, 1)
    """
    x1 = matrix[:, 0]
    y1 = matrix[:, 1]
    x2 = matrix[:, 2]
    y2 = matrix[:, 3]
    return x1, y1, x2, y2


def velocity_over_next_n_points(x: np.ndarray, y: np.ndarray, timestep: float, n_average: int) -> (float, float):
    """
    Calculate the average velocity vector over the next n timesteps from the given timestep
    :param  x:         array, N, of x coordinates
    :param  y:         array, N, of y coordinates
    :param  timestep:  timestep to calculate the velocity vector at
    :param  n_average: number of points to average over
    :return: (vx, vy): velocity unit vector
    """
    assert x.shape[0] == y.shape[0]
    assert timestep >= 0
    assert n_average > 0
    assert timestep + n_average < x.shape[0]

    # define xs but with better code
    xs = np.mean(x[timestep + 1:timestep + n_average + 1] - x[timestep])
    ys = np.mean(y[timestep + 1:timestep + n_average + 1] - y[timestep])
    norm = np.sqrt(xs**2 + ys**2)
    return xs/norm, ys/norm


def plot_trajectory_vectors(x: np.ndarray,
                            y: np.ndarray,
                            d: int,
                            n_average: int = 40,
                            start_color='darkgreen',
                            arrow_color='r',
                            arrow_scale=10,
                            zorder=2) -> None:
    """
    Plot the velocity vectors of the trajectory given by `x` and `y`
    :param x:               array, N, of x coordinates
    :param y:               array, N, of y coordinates
    :param d:               timestep difference between each arrow to draw
    :param n_average:       number of points to average over
    :param start_color:     color of the first arrow
    :param arrow_color:     color of the other arrows
    :param arrow_scale:     scale of the arrows (higher is smaller!)
    """
    assert x.shape[0] == y.shape[0] and x.shape == y.shape

    for time_step in range(0, x.shape[0] - n_average, d):
        this_color = arrow_color
        if time_step == 0:
            this_color = start_color
        avg_x, avg_y = velocity_over_next_n_points(x, y, time_step, n_average)
        # zorder = 2 to draw arrows on top of the trajectory
        plt.quiver(x[time_step], y[time_step], avg_x, avg_y, scale=arrow_scale, color=this_color, zorder=zorder)


def check_two_pedestrian_arrays(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> None:
    """
    Check that the two pedestrian arrays are the same length
    :param x1: array, N, of x coordinates of pedestrian 1
    :param y1: array, N, of y coordinates of pedestrian 1
    :param x2: array, N, of x coordinates of pedestrian 2
    :param y2: array, N, of y coordinates of pedestrian 2
    """
    assert x1.shape[0] == y1.shape[0] and x1.shape == y1.shape
    assert x2.shape[0] == y2.shape[0] and x2.shape == y2.shape
    assert x1.shape == x2.shape


