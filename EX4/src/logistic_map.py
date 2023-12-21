import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def logistic_step(x: float, r: float) -> float:
    """
    Performs a step of the logistic map.

    :param x: x_n.
    :param r: the bifurcation parameter r.
    :returns: x_(n+1) for the logistic map
    """
    return r * x * (1 - x)


def logistic_n_steps(
    *, x0: float, r: list[float], n: int, n_skip=0, plot: bool = False
) -> tuple[list[float], list[float]]:
    """
    Performs n steps of the logistic map for each value in r, and plots the trajectories.

    :param x0: the initial condition of the system.
    :param r: the bifurcation parameters r. it is a list of all the values of r required.
    :param n: the number of that will run for each r.
    :param n_skip: the number of iterations skipped at the start of each r
    :param plot: set to true if you want to plot the trajectory for each r
    :returns: A tuple. First value is a list of unskipped x values for each r it is of size
    of size (n - n_skip) * len(r). The second value is a list of r values for each x.
    """
    x_vals_all = []
    r_vals_all = []

    n_plots = len(r)
    rows = int(np.ceil(n_plots / 2))
    if plot:
        fig, axes = plt.subplots(rows, 2, figsize=(20, 4 * rows))

    for r_index, curr_r in enumerate(r):
        n_vals, x_vals = [], []
        x = x0
        for t in range(0, n + 1):
            if t > n_skip:
                x_vals.append(x)
                n_vals.append(t)

            x = logistic_step(x, curr_r)

        if plot:
            row = r_index // 2
            col = r_index % 2
            ax = axes[row, col] if n_plots > 2 else axes[col]
            ax.set_title(f"x0= {x0:.2f}, r= {curr_r:.2f}, n= {n}")
            ax.set_xlabel("t")
            ax.set_ylabel("x")
            ax.xaxis.set_ticks(np.arange(0, n + 1, 5.0))
            sns.set_style("darkgrid")
            ax.plot(n_vals, x_vals, c="black")

        x_vals_all.extend(x_vals)
        r_vals_all.extend([curr_r for _ in x_vals])

    if plot:
        plt.tight_layout()

    return r_vals_all, x_vals_all


def plot_logistic_bifurcation(x0: float, n: int, n_skip: int) -> None:
    """
    Plots the bifurcation diagram for the logistic function

    :param x0: the initial condition of the system.
    :param n: the number of that will run for each r.
    :param n_skip: the number of iterations skipped at the start of each r
    """
    R, X = logistic_n_steps(
        x0=x0, r=np.linspace(0.0, 4.0, 1000), n=n, n_skip=n_skip, plot=False
    )

    plt.figure(figsize=(16, 6), dpi=100)
    plt.title("Logistic Function bifurcation Diagram")
    plt.plot(R, X, c="black", ls="", marker=",", alpha=0.5)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 4.1, 0.1))
    plt.xlim(0, 4)
    plt.xlabel("r")
    plt.ylabel("x")
    plt.show()
