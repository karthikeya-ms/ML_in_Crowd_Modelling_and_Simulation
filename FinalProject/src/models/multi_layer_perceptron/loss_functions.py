"""Contains all code related to loss functions."""

import numpy as np

def cross_entropy_loss(y_pred, y):
    """
    Compute the cross-entropy loss for a classification task.

    Parameters:
    - y_pred (numpy.ndarray): Predicted probabilities.
    - y (numpy.ndarray): True labels (one-hot encoded).

    Returns:
    - float: Cross-entropy loss.
    """
    batch_size = y.shape[0]
    return -np.sum(np.log(y_pred) * y) / batch_size

def cross_entropy_loss_backwards(y_pred, y):
    """
    Compute the derivative of the cross-entropy loss.

    Parameters:
    - y_pred (numpy.ndarray): Predicted probabilities.
    - y (numpy.ndarray): True labels (one-hot encoded).

    Returns:
    - numpy.ndarray: Derivative of the cross-entropy loss.
    """
    return -np.sum(y / np.log(y_pred), axis=1)

def l2_regularization_term(parameters, batch_size, lambd):
    """
    Compute the L2 regularization term for the weights.

    Parameters:
    - parameters (dict): Model parameters containing weights.
    - batch_size (int): Size of the training batch.
    - lambd (float): Regularization parameter.

    Returns:
    - float: L2 regularization term.
    """
    W = parameters["weights"]
    L2 = 0
    for cur in W:
        L2 = np.squeeze(L2 + np.sum(np.square(W[cur])))
    L2 = (lambd / (2 * batch_size)) * L2

    return L2
