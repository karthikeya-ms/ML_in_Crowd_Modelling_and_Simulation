import numpy as np
import matplotlib.pyplot as plt


def plot_linear_fit(x: np.ndarray, y: np.ndarray, a: float, residual: float) -> None:
    """Plot linear fit.

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        a: Slope of the line.
        residual: Residuals of the linear fit.
    """
    plt.scatter(x, y)
    plt.plot(x, a * x, color="red", linewidth=1)
    plt.legend(["data", "least-squares fit"])
    plt.title(f"Slope: {a:.2f}, residual: {residual:.2f}")
    plt.show()
    return None


def linear_approximate_1d(x: np.ndarray, y: np.ndarray, plot: bool = True) -> tuple:
    """Linear approximate 1D data.

    Args:
        x:         1D array of x coordinates.
        y:         1D array of y coordinates.
        plot:      Whether to plot the linear fit.

    Returns:
        a:         Slope of the line.
        residuals: Residuals of the linear fit.

    """
    solution, residuals, rank, singular_values = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)
    assert rank == 1, "Only 1D data is supported."

    if plot:
        plot_linear_fit(x, y, solution[0], residuals[0])

    return solution, residuals


def linear_approximate_closed_form(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Approximates solution of the linear system X @ A.T = F using the closed form solution of the least-squares problem.

    Args:
        X: Array of domain coordinates.
        F: Array of target coordinates.

    Returns:
        A: Least-square solution.
    """
    F = F.reshape(-1, 1)
    X = X.reshape(-1, 1)
    return (np.linalg.inv(X.T @ X) @ X.T @ F).T


def get_extrema(x: np.ndarray, y: np.ndarray, k: int = 1, weight_function=None) -> np.ndarray:
    """
    Finds x values whose y value is extreme compared to the k average neighboring points.
    The parameter k is to make the result less sensitive to noise.
    Can use weighted average
    This could also be generalized to n dimensions using a suitable metric (like l2) to get the k nearest neighbors.

    Args:
        x:               1D array of domain coordinates.
        y:               1D array of target coordinates.
        k:               Number of neighbors to compare to on both directions.
        weight_function: Function that takes two points and returns a weight. If None, defaults to exponential weight.

    Returns:
        x_extrema:       1D array of x coordinates of points flagged as extrema.
    """
    assert k < len(x), "k must be smaller than the length of x."
    assert len(x) == len(y), "x and y must have the same length."

    weighted_average = True

    x_minima = []
    x_maxima = []
    for i in range(k, len(x) - k):
        x_lower = x[i - k: i]
        x_upper = x[i + 1: i + k + 1]
        y_lower = y[i - k: i]
        y_upper = y[i + 1: i + k + 1]
        lower_avg = np.mean(y_lower)
        upper_avg = np.mean(y_upper)

        # WEIGHTED CODE
        if weighted_average:
            weights_lower = np.exp(-np.abs(x_lower - x[i]))
            weights_upper = np.exp(-np.abs(x_upper - x[i]))
            lower_avg = np.sum(y_lower * weights_lower) / np.sum(weights_lower)
            upper_avg = np.sum(y_upper * weights_upper) / np.sum(weights_upper)
        # END WEIGHTED CODE

        # local maximum
        if y[i] > lower_avg and y[i] > upper_avg:
            # if any other point in the neighborhood is greater or equal to the current point, it is not a local maximum
            if np.any(y_lower >= y[i]) or np.any(y_upper >= y[i]):
                continue
            x_maxima.append(x[i])
        # local minimum
        elif y[i] < lower_avg and y[i] < upper_avg:
            # if any other point in the neighborhood is less or equal to the current point, it is not a local minimum
            if np.any(y_lower <= y[i]) or np.any(y_upper <= y[i]):
                continue
            x_minima.append(x[i])

    # one final pass to eliminate entries that are too close to each other

    RANGE_TO_CHECK = 2

    for i in range(RANGE_TO_CHECK * k, len(x) - RANGE_TO_CHECK * k):
        if x[i] in x_minima:
            for j in range(i - RANGE_TO_CHECK * k, i + RANGE_TO_CHECK * k + 1):
                if x[j] in x_minima:
                    if x[j] < x[i]:
                        x_minima.remove(x[j])
                    elif x[j] > x[i]:
                        x_minima.remove(x[i])
        if x[i] in x_maxima:
            for j in range(i - RANGE_TO_CHECK * k, i + RANGE_TO_CHECK * k + 1):
                if x[j] in x_maxima:
                    if x[j] < x[i]:
                        x_maxima.remove(x[i])
                    elif x[j] > x[i]:
                        x_maxima.remove(x[j])

    # Final tweak: add the first and last points regardless of whether they are extrema
    end_points = [np.max(x), np.min(x)]
    return np.array(x_minima + x_maxima + end_points)


def rbf_interpolate(x: np.ndarray, y: np.ndarray, epsilon: np.ndarray, l_array: np.ndarray) -> tuple:
    """Interpolate the data with radial basis functions.

    Args:
      x: x values of the data
      y: y values of the data
      epsilon: width of the peaks. This is the value of epsilon, not epsilon^2
      l_array: array of x values of the peaks

    Returns:
      y_hat: interpolated y values
    """

    # N: number of data points
    # L: number of peaks to use in the interpolation
    N, L = x.shape[0], l_array.shape[0]
    phi = np.zeros((N, L))
    for i in range(N):
        for j in range(L):
            phi[i, j] = np.exp(-((x[i] - l_array[j]) / epsilon) ** 2)
            # a suitable number for rcond is high
    c, residuals, rank, s = np.linalg.lstsq(phi, y, rcond=None)

    plot_rbf_interpolation(x, y, epsilon, l_array, c)
    return c, residuals, rank, s


def plot_rbf_interpolation(x, y, epsilon, l_vector, c) -> None:
    """Plot the data and the RBF interpolation.

    Args:
      x:        x values of the data
      y:        y values of the data
      epsilon:  width of the peaks. This is the value of epsilon, not epsilon^2
      l_array:  array of x values of the peaks
      c:        coefficients of the RBF interpolation
    """

    def get_y_plot(x_value):
        y_value = 0
        for i in range(c.shape[0]):
            y_value += c[i] * np.exp(-((x_value - l_vector[i]) / epsilon) ** 2)
        return y_value

    plt.scatter(x, y)
    min, max = np.min(x), np.max(x)
    x_plot = np.linspace(min, max, 1000)
    plt.plot(x_plot, get_y_plot(x_plot), color="red", linewidth=1)
    # Otherwise, we see too many and it obfuscates the plot
    if len(l_vector) < 100:
        plt.scatter(l_vector, get_y_plot(l_vector), marker="x", color="orange", zorder=5, s=100)
    plt.title(f'RBF interpolation with ε={epsilon}, L={l_vector.shape[0]}')
    plt.legend(["data", "RBF interpolation", "x_l values"])
    plt.show()


def get_extrema_tmp(x, y):
    """
    Ignore this
    """
    from scipy.signal import argrelextrema
    # return the x values of the extrema
    minima = x[argrelextrema(y, np.greater)[0]]
    maxima = x[argrelextrema(y, np.less)[0]]
    return np.concatenate((minima, maxima))
