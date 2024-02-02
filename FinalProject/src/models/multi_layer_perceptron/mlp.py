import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.data_helpers import random_mini_batches
from src.models.ml_model import MlModel
from src.models.multi_layer_perceptron.activations import Softmax
from src.models.multi_layer_perceptron.loss_functions import (
    cross_entropy_loss,
    cross_entropy_loss_backwards,
    l2_regularization_term,
)
from src.models.multi_layer_perceptron.optimizer import Adam


class MLP(MlModel):
    def __init__(
        self, *, input_feature_size, layer_dimensions, activations, config
    ) -> None:
        self.config = config
        self.layer_dimensions = layer_dimensions
        self.activations = activations
        self.parameters = self._initialize_parameters(input_feature_size)

    def train(self, X_train, y_train):
        # list of losses
        losses_train = []
        losses_val = []

        # optimizer initialization
        optimizer = Adam(
            self.parameters,
            self.config["lr"],
        )

        for current_epoch in range(self.config["epochs"]):
            # randomize mini batches
            mini_batches = random_mini_batches(
                X_train, y_train, self.config["batch_size"]
            )

            # update the learning rate
            optimizer.set_learning_rate(
                self.config["lr"] / (1 + current_epoch * self.config["lr_decay"])
            )

            batch_losses = 0.0
            with tqdm(mini_batches, unit="batch") as tepoch:
                for batch_index, (batch_X, batch_y) in enumerate(tepoch):
                    tepoch.set_description(
                        f"[Epoch {current_epoch + 1}][Batch {batch_index + 1}/{len(tepoch)}]"
                    )

                    # forward propagation
                    batch_y_pred = self(batch_X)

                    # loss
                    loss = cross_entropy_loss(batch_y_pred, batch_y)

                    # keeping track of losses
                    batch_losses += loss

                    loss += l2_regularization_term(
                        self.parameters,
                        self.config["batch_size"],
                        self.config["regularization_parameter"],
                    )

                    # back propagation
                    grads = self._back_propagation(batch_y)

                    # update parameters
                    self.parameters = optimizer.update_parameters(
                        self.parameters, grads
                    )

                    tepoch.set_postfix(loss=batch_losses / (batch_index + 1))

            epochs_loss = batch_losses / len(mini_batches)
            losses_train.append(epochs_loss)

            X_val, y_val = self.config["validation_set"]
            y_val_pred = self(X_val)
            validation_loss = cross_entropy_loss(y_val_pred, y_val)
            losses_val.append(validation_loss)

        # plot the cost
        plt.plot(losses_train, label="Training")
        plt.plot(losses_val, label="Validation")
        plt.legend(loc="upper right")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training/Val Loss Curve")
        plt.show()

    def _initialize_parameters(self, input_feature_size):
        rng = np.random.default_rng()

        # L = number of layers (not counting input)
        L = len(self.layer_dimensions)

        # insert the input layer size to make things easier
        self.layer_dimensions.insert(0, input_feature_size)
        weights = {}
        biases = {}

        # Xavier Initialization
        for i in range(1, L + 1):
            weights["W" + str(i)] = rng.standard_normal(
                (self.layer_dimensions[i - 1], self.layer_dimensions[i])
            ) * np.sqrt(1.0 / self.layer_dimensions[i - 1])
            biases["b" + str(i)] = np.zeros((1, self.layer_dimensions[i]))

        return {"weights": weights, "biases": biases}

    def _back_propagation(self, y):
        W = self.parameters["weights"]
        L = len(W)
        n = y.shape[0]

        grads = {}
        dA = 0.0  # placeholder value, real value is set in first iteration
        for i in reversed(range(L)):
            # ==========dZ===========
            cross_entropy_loss_backwards(self.cache["A" + str(L)], y)

            # trick for softmax loss and cross entropy loss derivative. Not generalizable to other models
            if isinstance(self.activations[i], Softmax):
                dZ = self.cache["A" + str(L)] - y
            else:
                dZ = dA * self.activations[i].backwards(self.cache["Z" + str(i + 1)])

            # ==========dW===========
            dW = (1 / n) * (
                np.dot(self.cache["A" + str(i)].T, dZ)
                + self.config["regularization_parameter"] * W["W" + str(i + 1)]
            )
            # ==========db===========
            db = (1 / n) * np.sum(dZ, axis=0, keepdims=True)
            # ==========dA===========
            dA = np.dot(dZ, W["W" + str(i + 1)].T)
            # ======gradients========
            grads["dW" + str(i + 1)] = dW
            grads["db" + str(i + 1)] = db

        return grads

    def save_parameters(self, path):
        # save the learned parameters
        file = open(path, "wb")
        pickle.dump(self.parameters, file)
        file.close()

    def load_parameters(self, path):
        params = pickle.load(open(path, "rb"))
        self.parameters = params

    # normal forward prop
    def __call__(self, X):
        L = len(self.parameters["weights"])
        W = self.parameters["weights"]
        b = self.parameters["biases"]
        self.cache = {"A0": X}

        # each perceptron calculates a linear function (WX + b)
        # followed by an activation function to introduce non-linearity
        # to the network
        for i in range(1, L + 1):
            # Linear activations
            self.cache["Z" + str(i)] = (
                np.dot(self.cache["A" + str(i - 1)], W["W" + str(i)]) + b["b" + str(i)]
            )
            # Non-linearity
            self.cache["A" + str(i)] = self.activations[i - 1](self.cache["Z" + str(i)])

        return self.cache["A" + str(L)]
