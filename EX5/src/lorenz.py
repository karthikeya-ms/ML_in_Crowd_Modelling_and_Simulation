import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pandas as pd


def lorenz(p0, t, sigma, beta, rho):
    """
    Calculate the lorenz ODEs for each axis

    :param p0: the initial condition of the system.
    :param t: the time steps of the system.
    :param sigma: the sigma parameter for the system
    :param beta: the beta parameter for the system
    :param rho: the rho parameter for the system
    :returns: dx, dy, dz of the system
    """
    x, y, z = p0
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return dx, dy, dz


def solve_lorenz(
    p0=[10, 10, 10],
    t=np.arange(0.0, 1000, 0.01),
    sigma=10,
    beta=8 / 3,
    rho=28,
):
    """
    Solve the Lorenz ODEs and get the 3d points for each time step

    :param p0: the initial condition of the system.
    :param t: the time steps of the system.
    :param sigma: the sigma parameter for the system.
    :param beta: the beta parameter for the system.
    :param rho: the rho parameter for the system.
    :returns: a dataframe of the 3d trajectory
    """
    sol = odeint(lorenz, p0, t, args=(sigma, beta, rho))
    xs, ys, zs = sol.T
    return pd.DataFrame({"x": xs, "y": ys, "z": zs})


def plot_lorenz(traj, title, xlabel, ylabel, zlabel):
    """
    Plots a lorenz attractor in 3d space

    :param traj: the trajectory of the attractor
    :param title: title of the plot
    :param xlabel: label of the x axis
    :param ylabel: label of the y axis
    :param zlabel: label of the z axis
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection="3d")

    ax.plot(traj.iloc[:, 0], traj.iloc[:, 1], traj.iloc[:, 2], linewidth=0.1, c="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
