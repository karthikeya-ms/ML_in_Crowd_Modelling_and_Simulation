import numpy as np
import matplotlib.pyplot as plt
import math


def plot_random_mnist_samples(X, n):
    X_size = len(X)
    images = X.reshape(-1, 28, 28)

    rng = np.random.default_rng()
    indices = rng.choice(range(X_size), size=n)
    samples = images[indices]

    ncols = 5
    nrows = math.ceil(n / 5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 4))
    fig.suptitle(f"{n} Random MNIST Samples")
    for img, ax in zip(samples, axes.flat):
        ax.set_axis_off()
        ax.imshow(img)

    for ax in axes.flat[n:]:
        ax.remove()
