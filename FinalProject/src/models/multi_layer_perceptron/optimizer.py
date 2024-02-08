"""Contains code related to optimizers."""

import numpy as np


class Adam:
    def __init__(self, model_parameters, lr=0.01, beta1=0.9, beta2=0.999) -> None:
        """
        Initialize the Adam optimizer.

        Parameters:
        - model_parameters (dict): Model parameters containing weights and biases.
        - lr (float): Learning rate (default is 0.01).
        - beta1 (float): Exponential decay rate for the first moment estimate (default is 0.9).
        - beta2 (float): Exponential decay rate for the second moment estimate (default is 0.999).
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        W = model_parameters["weights"]
        b = model_parameters["biases"]
        L = len(W)
        Mt = {}
        Vt = {}

        # for each layer
        for i in range(L):
            Mt["dW" + str(i + 1)] = np.zeros(W["W" + str(i + 1)].shape)
            Mt["db" + str(i + 1)] = np.zeros(b["b" + str(i + 1)].shape)

            Vt["dW" + str(i + 1)] = np.zeros(W["W" + str(i + 1)].shape)
            Vt["db" + str(i + 1)] = np.zeros(b["b" + str(i + 1)].shape)

        self.Mt = Mt
        self.Vt = Vt
        self.t = 1

    def update_parameters(self, model_parameters, gradients):
        """
        Update the model parameters using the Adam optimization algorithm.

        Parameters:
        - model_parameters (dict): Model parameters containing weights and biases.
        - gradients (dict): Gradients of the model parameters.

        Returns:
        - dict: Updated model parameters.
        """
        W = model_parameters["weights"]
        b = model_parameters["biases"]
        L = len(W)

        # learning rate for this iteration of Adam
        learning_rate_t = self.lr * (
            np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        )

        for i in range(L):
            # momentum
            self.Mt["dW" + str(i + 1)] = (
                self.beta1 * self.Mt["dW" + str(i + 1)]
                + (1 - self.beta1) * gradients["dW" + str(i + 1)]
            )
            self.Mt["db" + str(i + 1)] = (
                self.beta1 * self.Mt["db" + str(i + 1)]
                + (1 - self.beta1) * gradients["db" + str(i + 1)]
            )

            # RMS
            self.Vt["dW" + str(i + 1)] = self.beta2 * self.Vt["dW" + str(i + 1)] + (
                1 - self.beta2
            ) * np.square(gradients["dW" + str(i + 1)])

            self.Vt["db" + str(i + 1)] = self.beta2 * self.Vt["db" + str(i + 1)] + (
                1 - self.beta2
            ) * np.square(gradients["db" + str(i + 1)])

            W["W" + str(i + 1)] = W["W" + str(i + 1)] - learning_rate_t * (
                self.Mt["dW" + str(i + 1)]
                / (np.sqrt(self.Vt["dW" + str(i + 1)]) + 1e-8)
            )

            b["b" + str(i + 1)] = b["b" + str(i + 1)] - learning_rate_t * (
                self.Mt["db" + str(i + 1)]
                / (np.sqrt(self.Vt["db" + str(i + 1)]) + 1e-8)
            )
        self.t += 1
        return {"weights": W, "biases": b}

    def set_learning_rate(self, learning_rate):
        """
        Set the learning rate for the optimizer.

        Parameters:
        - learning_rate (float): New learning rate.
        """
        self.lr = learning_rate
