import numpy as np


def cross_entropy_loss(y_pred, y):
    batch_size = y.shape[0]
    return -np.sum(np.log(y_pred) * y) / batch_size


def cross_entropy_loss_backwards(y_pred, y):
    return -np.sum(y / np.log(y_pred), axis=1)


def l2_regularization_term(parameters, batch_size, lambd):
    W = parameters["weights"]
    L2 = 0
    for cur in W:
        L2 = np.squeeze(L2 + np.sum(np.square(W[cur])))
    L2 = (lambd / (2 * batch_size)) * L2

    return L2
