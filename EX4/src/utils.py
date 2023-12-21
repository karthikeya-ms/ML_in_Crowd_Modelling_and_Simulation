import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import Symbol, core
from typing import Tuple
from sympy.solvers import solve



def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k-1, :] + step_size * f_ode(yt[k-1, :])
    return yt, time


def plot_phase_portrait(A, X, Y):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Streamplot for linear vector field A*x');
    ax0.set_aspect(1)
    return ax0

def plot_bifurcation_diagram_1D(fun: str, min_alpha=-2, max_alpha=2.1, alpha_step=0.1):
    """
    Plots the bifurcation diagram of a given function to solve
    :param fun: string to represent the function to solve, based on having 'x' as variable
    """
    x = Symbol('x')
    alphas = np.arange(min_alpha, max_alpha, alpha_step)
    alphas = [round(alpha, 7) for alpha in alphas]
    fixed_points = {}
    fixed_points_rel_alphas = {}
    for alpha in alphas:
        sol = solve(eval(fun), x)
        for i, single_sol in enumerate(sol):
            if i not in fixed_points:
                fixed_points[i] = [single_sol]
                fixed_points_rel_alphas[i] = [alpha]
            else:
                fixed_points[i].append(single_sol)
                fixed_points_rel_alphas[i].append(alpha)
    # postprocessing
    for i in sorted(fixed_points.keys()):
        for j in range(len(fixed_points[i])):
            if not isinstance(fixed_points[i][j], core.numbers.Float) and not isinstance(fixed_points[i][j],
                                                                                         core.numbers.Integer):
                fixed_points[i][j] = None
        plt.scatter(fixed_points_rel_alphas[i], fixed_points[i])
    plt.xlim(alphas[0], alphas[-1])
    print(alphas[0])
    # plt.ylim(-1, 1)
    plt.title(fun)
    plt.show()