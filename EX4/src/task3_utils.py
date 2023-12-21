import numpy as np
import matplotlib.pyplot as plt


def get_orbit_from(x_0, n_points = 1000):
    def polar_to_cartesian(rho, phi):
        return rho*np.cos(phi), rho*np.sin(phi)
    delta_t = 0.01
    rho = np.zeros(n_points)
    phi = np.zeros(n_points)
    rho[0] = np.sqrt(x_0[0] ** 2 + x_0[1] ** 2)
    # make sure this works for points on the real X axis
    phi[0] = np.arctan2(x_0[1], x_0[0])
    for i in range(1, n_points):
        rho[i] = rho[i-1] + delta_t*(rho[i-1]*(1 - rho[i-1]**2))
        # NOTE: Yes, phi can overflow beyond 2pi, but it doesn't matter
        phi[i] = phi[i-1] + delta_t
    return polar_to_cartesian(rho, phi)


def plot_phase_diagram(alpha, axis_range, n_points=1000, plot_orbit_start=None):
    """Plots the Andronov-Hopf system's phase diagram for the given alpha value."""
    x1 = np.linspace(axis_range[0], axis_range[1], n_points)
    x2 = np.linspace(axis_range[0], axis_range[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    # Our velocities
    r_squared = np.square(X1) + np.square(X2)
    X1_dot = alpha*X1 - X2 - X1*r_squared
    X2_dot = X1 + alpha*X2 - X2*r_squared
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.streamplot(X1, X2, X1_dot, X2_dot, density=1, color='k')

    title_string = f'α = {alpha}'
    if alpha > 0:
        rho_0 = np.sqrt(alpha)
        title_string += f', ρ₀ ≈ {rho_0:.2f}'
        # Plot the stable orbit in purple
        ax.plot(rho_0*np.cos(x1), rho_0*np.sin(x1), 'blue', zorder=2)
        ax.plot(-rho_0*np.cos(x1), -rho_0*np.sin(x1), 'blue', zorder=2)

    # Plot our orbit
    if plot_orbit_start is not None:
        x_orbit, y_orbit = get_orbit_from(plot_orbit_start)
        ax.plot(x_orbit, y_orbit, 'red', zorder=3)
        # plot the start and end points of the orbit
        ax.plot(x_orbit[0], y_orbit[0], 'o', color='red', zorder=4)
        ax.plot(x_orbit[-1], y_orbit[-1], 'o', color='green', zorder=4)

    ax.set_title(title_string)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    plt.show()


def plot_cusp_surface():
    """
    Plot the cusp bifurcation surface, (X,Y,Z) = (alpha_1, alpha_2, x)
    Return figure to replot with different angles
    """
    # New figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def cusp_normal_form(a, b, c):
        return a+b*c+a**3

    n = 50
    A = np.linspace(-n, n, 100)

    # Create a grid to plot the surface
    A_grid = np.meshgrid(A,A)

    # Plot the surface for the cusp normal form

    # X plane
    for x in A:
        Y,Z = A_grid[0], A_grid[1]
        X = cusp_normal_form(x,Y,Z)
        ax.contour(X+x, Y, Z, [x], zdir='x', colors='darkblue')

    # Y plane
    for y in A:
        X,Z = A_grid[0], A_grid[1]
        Y = cusp_normal_form(X,y,Z)
        ax.contour(X, Y+y, Z, [y], zdir='y', colors='darkblue')

    # Z plane
    for z in A:
        X,Y = A_grid[0], A_grid[1]
        Z = cusp_normal_form(X,Y,z)
        ax.contour(X, Y, Z+z, [z], zdir='z', colors='darkblue')

    # Plot limits.
    ax.set_zlim3d(-n, n)
    ax.set_xlim3d(-n, n)
    ax.set_ylim3d(-n, n)
    ax.set_xlabel('α₁')
    ax.set_ylabel('α₂')
    ax.set_zlabel('x')
    plt.show()
    return fig, ax
