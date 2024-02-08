import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_diabetes


def load_mnist(*, validation_size, OHE=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=7
    )

    if OHE:
        ohe = OneHotEncoder()
        y_train = ohe.fit_transform(y_train).toarray()
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
