"""Contains the functions concerned with data loading and manipulation."""

import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_diabetes


def load_mnist(*, validation_size, OHE=False):
    """
    Load the MNIST dataset and preprocess it for training.

    Parameters:
    - validation_size (float): Proportion of the training data to use for validation. Should be in the range [0, 1],
    validation set will be None if this value is 0.
    - OHE (bool): Whether to perform one-hot encoding on the labels. Default is False.

    Returns:
    - X_train (numpy.ndarray): Training features normalized between 0 and 1.
    - y_train (numpy.ndarray or None): Training labels, optionally one-hot encoded if OHE is True.
    - X_val (numpy.ndarray or None): Validation features, only returned if validation_size > 0.
    - y_val (numpy.ndarray or None): Validation labels, optionally one-hot encoded if OHE is True.
    - X_test (numpy.ndarray): Test features normalized between 0 and 1.
    - y_test (numpy.ndarray or None): Test labels, optionally one-hot encoded if OHE is True.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    X_val = None
    y_val = None
    if validation_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=7
        )

    if OHE:
        ohe = OneHotEncoder()
        y_train = ohe.fit_transform(y_train).toarray()
        if y_val is not None:
            y_val = ohe.transform(y_val).toarray()
        y_test = ohe.transform(y_test).toarray()

    return X_train, y_train, X_val, y_val, X_test, y_test

def forrester_function(x):
    return np.multiply(np.power((6*x-2),2), np.sin(12*x-4))

def load_forrester(*, begin: float, end: float, noise_variance: float, n_samples: int, validation_size: float =0.2):
    X = np.random.random(size=n_samples) * (end - begin) + begin
    y = forrester_function(X)

    std_dev = np.sqrt(noise_variance)
    for i, target in enumerate(y):
        y[i] = np.random.normal(target, std_dev)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_size, random_state=7
    )

    return X_train, y_train, X_test, y_test


def load_diabetes_sklearn(*, validation_size, random=False):
    dataset = load_diabetes()

    if random:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset['data'], dataset['target'], test_size=validation_size
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset['data'], dataset['target'], test_size=validation_size, random_state=7
        )

    return X_train, y_train, X_test, y_test


def random_mini_batches(X, y, batch_size):
    """
    Generate random mini-batches from the input data for stochastic gradient descent.

    Parameters:
    - X (numpy.ndarray): Input features.
    - y (numpy.ndarray): Corresponding labels.
    - batch_size (int): Size of each mini-batch.

    Returns:
    - mini_batches (list): List of tuples containing mini-batches (X_mini, y_mini).
      Each tuple represents a mini-batch, and X_mini and y_mini are the features and labels, respectively.
    """
    rng = np.random.default_rng()
    mini_batches = []
    n = X.shape[0]

    # randomize the set
    permutation = list(rng.permutation(n))
    X = X[permutation, :]
    y = y[permutation, :]

    # number of mini batches (-1 if there is a remainder)
    mini_batch_num = n // batch_size

    # divide the set into mini batches
    for i in range(mini_batch_num):
        MX = X[i * batch_size : (i + 1) * batch_size, :]
        MY = y[i * batch_size : (i + 1) * batch_size, :]
        mini_batches.append((MX, MY))

    # last mini batch
    if n % batch_size != 0:
        MX = X[mini_batch_num * batch_size :, :]
        MY = y[mini_batch_num * batch_size :, :]
        mini_batches.append((MX, MY))

    return mini_batches
