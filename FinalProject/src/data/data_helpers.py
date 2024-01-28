import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_mnist(*, test_size, validation_size, OHE=False):
    # X.shape = (1797, 64), y.shape = (1797,)
    X, y = load_digits(return_X_y=True)
    X /= 255
    y = y.reshape(-1, 1)

    if OHE:
        ohe = OneHotEncoder()
        y = ohe.fit_transform(y).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=7
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=7
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def random_mini_batches(X, y, batch_size):
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
