import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def lorenz(p0, t, sigma, beta, rho):
    x, y, z = p0
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return dx, dy, dz


def solve_lorenz(
    p0=[10, 10, 10],
    x_perturb=0.0,
    t=np.arange(0.0, 1000, 0.01),
    sigma=10,
    beta=8 / 3,
    rho=28,
    plot=True,
    ax=None,
):
    p0_perturb = (p0[0] + x_perturb, p0[1], p0[2])
    sol = odeint(lorenz, p0_perturb, t, args=(sigma, beta, rho))
    xs, ys, zs = sol.T

    if plot:
        if ax is None:
            fig = plt.figure(figsize=(16, 10), dpi=400)
            ax = fig.add_subplot(projection="3d")

        ax.plot(xs, ys, zs, linewidth=0.1, c="black")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        (start_handle,) = ax.plot(
            [xs[0]], [ys[0]], [zs[0]], "o", ms=5, color="green", alpha=1
        )
        (end_handle,) = ax.plot(
            [xs[-1]], [ys[-1]], [zs[-1]], "o", ms=5, color="red", alpha=1
        )
        start_point_str = (
            f"Start Point = ({p0[0]:.2f} + {x_perturb:.0E}, {ys[0]:.2f}, {zs[0]:.2f})"
            if x_perturb > 0
            else f"Start Point = ({p0[0]:.2f}, {ys[0]:.2f}, {zs[0]:.2f})"
        )
        end_point_str = f"End Point = ({xs[-1]:.2f}, {ys[-1]:.2f}, {zs[-1]:.2f})"
        ax.legend([start_handle, end_handle], [start_point_str, end_point_str])

    return xs, ys, zs, t


def plot_trajectory_diff(
    xs_1, ys_1, zs_1, xs_2, ys_2, zs_2, t, ax=None, title="", horizontal=None
):
    points_1 = np.vstack([xs_1, ys_1, zs_1]).T
    points_2 = np.vstack([xs_2, ys_2, zs_2]).T
    dist = np.linalg.norm(points_1 - points_2, axis=1)

    if ax == None:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot()

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("difference")
    ax.plot(t, dist, c="black")
    if horizontal is not None:
        handle = ax.axhline(y=horizontal, color="r", linestyle="-")
        ax.legend([handle], [f"y = {horizontal:.2f}"])

    return dist


def plot_lorenz_bifurcation_rho(
    p0=[10, 10, 10],
    t=np.arange(0.0, 1000, 0.01),
    n_skip=9900,
    sigma=10,
    beta=8 / 3,
    rho=np.arange(0.5, 28.5, 0.5),
):
    xs_all, ys_all, zs_all, rho_all = [], [], [], []
    for curr_rho in rho:
        xs, ys, zs, ts = solve_lorenz(
            p0=p0, t=t, sigma=sigma, beta=beta, rho=curr_rho, plot=False
        )
        xs, ys, zs, ts = (
            xs[n_skip + 1 :],
            ys[n_skip + 1 :],
            zs[n_skip + 1 :],
            ts[n_skip + 1 :],
        )

        xs_all.extend(list(xs))
        ys_all.extend(list(ys))
        zs_all.extend(list(zs))
        rho_all.extend([curr_rho for _ in range(len(xs))])

    fig = plt.figure(figsize=(16, 6), dpi=100)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(rho_all, xs_all, c="black", ls="", marker=",", alpha=0.5)
    ax1.set_xlabel("rho")
    ax1.set_ylabel("x")
    ax1.set_title("Lorenz System bifurcation Diagram W.R.T. Rho (X axis)")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(rho_all, ys_all, c="black", ls="", marker=",", alpha=0.5)
    ax2.set_xlabel("rho")
    ax2.set_ylabel("y")
    ax2.set_title("Lorenz System bifurcation Diagram W.R.T. Rho (Y axis)")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(rho_all, zs_all, c="black", ls="", marker=",", alpha=0.5)
    ax3.set_xlabel("rho")
    ax3.set_ylabel("z")
    ax3.set_title("Lorenz System bifurcation Diagram W.R.T. Rho (Z axis)")
    fig.tight_layout()
