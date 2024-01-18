import pandas as pd
import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Union, Iterable, Tuple
from scipy.spatial.distance import cdist


def read_vectorfield_data(dir_path="../data/task_3/", base_filename="nonlinear_vectorfield_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the 2 files containing the vector field data
    :param dir_path: path of the directory containing the 2 files
    :param base_filename: common part of the name in the 2 files, then the suffix "_x0.txt" or "_x1.txt" is added
    :returns: the data contained in the 2 files in the form of 2 numpy ndarrays
    """
    x0 = pd.read_csv(dir_path + base_filename + "_x0.txt", sep=' ', header=None).to_numpy()
    x1 = pd.read_csv(dir_path + base_filename + "_x1.txt", sep=' ', header=None).to_numpy()
    return x0, x1


def estimate_vectors(delta_t: float, x0=None, x1=None) -> np.ndarray:
    """
    Estimates the vector field using the finite-difference formula
    :param delta_t: the time difference used as denominator of the time-difference formula
    :param x0: the data at beginning of time delta
    :param x1: the data at end of time delta
    :returns: an approximation of the vectors s.t. v(x0_k) = x1_k
    """
    # read the 2 files containing the vector field data (if data is not given)
    if x0 is None or x1 is None:
        x0, x1 = read_vectorfield_data()
    # estimate the vector field through the finite-difference formula
    vectors = (x1 - x0) / delta_t
    return vectors


def create_phase_portrait_matrix(A: np.ndarray, title_suffix: str, save_plots=False,
                                 save_path: str = None, display=True):
    """
    Plots the phase portrait of the linear system Ax
    :param A: system's (2x2 matrix in our case)
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    """
    w = 10  # width
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of A: ", eigenvalues)
    # linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(10, 10))
    plt.streamplot(X, Y, U, V, density=1.0)
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)


def solve_trajectory(x0, x1, funct, args, find_best_dt=False, end_time=0.1, plot=False):
    """
    Solves initial value point problem for a whole dataset of points, up to a certain moment in time
    :param x0: the data at time 0
    :param x1: the data at unknown time step after 0
    :param funct: to get derivative for next steps generation
    :param find_best_dt: if True also the dt where we have lowest MSE is searched
    :param end_time: end time for the simulation
    :param plot: boolean to produce a scatter plot of the trajectory (orange) with the final x1 points in blue
    :returns: points at time end_time, best point in time (getting lowest MSE), lowest MSE
    """
    # initialize variables for find_best_dt procedure
    best_dt = -1
    best_mse = math.inf
    x1_pred = []
    # fixate some times where system must be evaluated
    t_eval = np.linspace(0, end_time, 100)
    sols = []
    for i in range(len(x0)):
        sol = solve_ivp(funct, [0, end_time], x0[i], args=args, t_eval=t_eval)  # solve initial value problem for a given point
        x1_pred.append([sol.y[0, -1], sol.y[1, -1]])  # save the final solution
        if find_best_dt:
            # to find best dt then all the different snapshots in time have to be saved
            sols.append(sol.y)
        # plot the trajectory (orange) and ground truth end point (blue)
        if plot:
            plt.scatter(x1[i, 0], x1[i, 1], c='blue', s=10)
            plt.scatter(sol.y[0, :], sol.y[1, :], c='orange', s=4)
    if find_best_dt:
        # try all the different moments in time, check if it is the best time
        for i in range(len(t_eval)):
            pred = [[sols[el][0][i], sols[el][1][i]] for el in range(len(sols))]
            mse = np.mean(np.linalg.norm(pred - x1, axis=1)**2)
            # if mse found is best yet, update the variables
            if mse < best_mse:
                best_mse = mse
                best_dt = t_eval[i]
    if plot:
        plt.rcParams["figure.figsize"] = (14,14)
        plt.show()
    return x1_pred, best_dt, best_mse


def find_best_rbf_configuration(x0, x1, dt=0.1, end_time=0.5):
    """
    grid search over various different eps and n_bases values, returning the whole configuration with lowest MSE
    :param x0: data at time 0
    :param x1: data after a certain unknown dt
    :param dt: dt to approximate the vector field between x0 and x1
    :param end_time: total time of solve_ivp system solving trajectory
    :return: best mse found with the configuration, including eps, n_bases, dt at which the mse was found, centers
    """
    final_best_mse, final_best_eps, final_best_n_bases, final_best_dt = math.inf, -1, -1, -1  # initialize variables
    n_bases_trials = [int(i) for i in np.linspace(100, 1001, 20)]  # define search space for n_bases
    for n_bases in n_bases_trials:
        centers = x0[np.random.choice(range(x0.shape[0]), replace=False, size=n_bases)]  # define centers
        for eps in (0.3, 0.5, 0.7, 1.0, 5.0, 10.0, 20.0):
            v = estimate_vectors(dt, x0, x1)  # estimate vector field
            C, res, _, _, _, eps, phi = approx_nonlin_func(data=(x0,v), n_bases=n_bases, eps=eps, centers=centers)
            x1_pred, best_dt, best_mse = solve_trajectory(x0, x1, rbf_approx, find_best_dt=True, args=[centers, eps, C], end_time=end_time, plot=False)
            if final_best_mse > best_mse:  # if new mse is better then update all return variables
                final_best_mse, final_best_eps, final_best_n_bases, final_best_dt, final_centers  = best_mse, eps, n_bases, best_dt, centers
    print(f"Printing best configuration: eps = {final_best_eps} - n_bases = {final_best_n_bases} - dt = {final_best_dt} giving MSE = {final_best_mse}")
    return final_best_mse, final_best_eps, final_best_n_bases, final_best_dt, final_centers


def create_phase_portrait_derivative(funct, args, title_suffix: str, save_plots=False,
                                     save_path: str = None, display=True, fig_size=10, w=4.5):
    """
    Plots the phase portrait given a 'funct' that gives the derivatives for a certain point
    :param funct: given a 2d point gives back the 2 derivatives
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    :param fig_size: gives width and height of plotted figure
    :param w: useful for defining range for setting Y and X
    """
    # setting up grid width/height
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # dynamic system parameter, responsible for the change in behaviour
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]:
            res = funct(0, np.array([x1, x2]), *args)
            U.append(res[0][0])
            V.append(res[0][1])
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    plt.figure(figsize=(fig_size, fig_size))
    plt.streamplot(X, Y, U, V, density=2, color='green')
    plt.title(f"{title_suffix}")
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)


def get_points_and_targets(data: Union[str, Iterable[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Depending on the type of the parameter 'data', returns correctly the points and the targets
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays: points and targets
    :returns: points and targets
    """
    if isinstance(data, str):
        data_path = data
        # read data
        linear_func_data = pd.read_csv(data_path, sep=" ", header=None, dtype=np.float64)
        # divide data into auxiliary variables
        points, targets = linear_func_data.iloc[:, 0], linear_func_data.iloc[:, 1]
        points = np.expand_dims(points, 1)  # add 1 dimension, needed for np.linalg.lstsq
    else:
        if len(data) != 2:
            raise ValueError(f"Parameter data must be either a string or an Iterable of 2 numpy ndarrays, got {len(data)} elements")
        points, targets = data[0], data[1]
    return points, targets


def rbf(x, x_l, eps):
    """
    radial basic function
    :param x: point/s
    :param x_l: center/s
    :param eps: radius of gaussians
    :return: matrix contains radial basic function
    """
    return np.exp(-cdist(x, x_l) ** 2 / eps ** 2)


def compute_bases(points: np.ndarray, eps: float, n_bases: int, centers: np.ndarray = None):
    """
    Compute the basis functions
    :param points: the points on which to calculate the basis functions
    :param centers: the center points to pick to compute the basis functions
    :param eps: epsilon param of the basis functions
    :param n_bases: number of basis functions to compute
    :returns: list of basis functions evaluated on every point in 'points'
    """
    if centers is None:
        # create n_bases basis functions' center points
        # centers = points[np.random.choice(points.ravel(), replace=False, size=n_bases)]
        centers = points[np.random.choice(range(points.shape[0]), replace=False, size=n_bases)]
    phi = rbf(points, centers, eps)
    return phi, centers


def approx_lin_func(data: Union[str, Iterable[np.ndarray]] = "../data/linear_function_data.txt") -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Approximate a linear function through least squares
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays: points and targets
    :returns: tuple (least squares solution, residuals, rank of coefficients matrix, singular values of coefficient matrix)
    """
    # get coefficients and targets from data
    points, targets = get_points_and_targets(data)
    # solve least square
    sol, residuals, rank, singvals = np.linalg.lstsq(a=points, b=targets, rcond=1e-5)
    return sol, residuals, rank, singvals


def approx_nonlin_func(data: Union[str, Iterable[np.ndarray]] = "../data/nonlinear_function_data.txt", n_bases: int = 5, eps: float = 0.1,
                       centers: np.ndarray = None):
    """
    Approximate a non-linear function through least squares
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays: points and targets
    :param n_bases: the number of basis functions to approximate the nonlinear function
    :param eps: bandwidth of the basis functions
    :param centers: list of center points to compute the basis functions
    :returns: tuple (least squares solution (transposed), residuals, rank of coefficients matrix, singular values of coefficient matrix, 
                    centers, eps and phi (list_of_basis))
    """
    # get coefficients and targets form the data
    points, targets = get_points_and_targets(data)

    # evaluate the basis functions on the whole data and putting each basis' result in an array
    list_of_bases, centers = compute_bases(points=points, centers=centers, eps=eps, n_bases=n_bases)

    # solve least square using the basis functions in place of the coefficients to use linear method with nonlinear function
    sol, residuals, rank, singvals = np.linalg.lstsq(a=list_of_bases, b=targets, rcond=1e-5)
    return sol, residuals, rank, singvals, centers, eps, list_of_bases


def plot_func_over_data(lstsqr_sol: np.ndarray, data: Union[str, Iterable[np.ndarray]], linear: bool, centers=None, eps=None, **kwargs):
    """
    Plot the approximated function over the actual data, given the solution of the least squares problem and the data
    :param lstsqr_sol: solution of the least squares problem
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays: points and targets
    :param linear: if True, plots the linear approximated function, otherwise the non-linear one
    :param centers: (optional) list of center points to compute the basis functions in case linear=False
    :param eps: (optional) epsilon parameter to compute the basis functions in case linear=False
    :param kwargs: (optional) can contain more data to include in the title of the plot, e.g. MSE of the approximation
    """
    plot_title = "Approximated function plotted over the actual data"

    # get the data's coefficients and targets
    points, targets = get_points_and_targets(data)

    # compute approximated function for every point on the x axis
    x = np.linspace(start=-5, stop=5, num=100)  # x axis
    if linear:
        y = lstsqr_sol * x  # y value for each x, used to plot the approximated data
    else:
        list_of_bases, centers = compute_bases(points=np.expand_dims(x, 1), centers=centers, eps=eps, n_bases=len(centers))
        y = np.sum(lstsqr_sol * list_of_bases, axis=1)  # '*' indicates and elementwise product (dimensions broadcast to common shape)
        plot_title += f"\nn_bases: {len(centers)}, eps: {eps}"

    # add eventual more data to the plot title
    for k, v in kwargs.items():
        plot_title += f", {k}: {v}"

    # plot approximated function over the actual data
    plt.figure(figsize=(5, 5))
    plt.scatter(points, targets, label="Data")
    plt.plot(x, y, color='r', label="Approximated function")
    plt.legend()
    plt.title(plot_title)
    plt.tight_layout()
    plt.show()
    
# Functions for solve_ivp
def rbf_approx(t, y, centers, eps, C):
    """
    function to return vector field of a single point (rbf)
    :param t: time (for solve_ivp)
    :param y: single point
    :param centers: all centers
    :param eps: radius of gaussians
    :param C: coefficient matrix, found with least squares
    :return: derivative for point y
    """
    y = y.reshape(1, y.shape[-1])
    phi = np.exp(-cdist(y, centers) ** 2 / eps ** 2)
    return phi @ C


def linear_approx(t, y, A):
    """
    function to return vector field of a single point (linear)
    :param t: time (for solve_ivp)
    :param y: single point
    :param A: coefficient matrix, found with least squares
    :return: derivative for point y
    """
    return A @ y