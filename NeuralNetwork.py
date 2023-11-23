import matplotlib.pyplot as plt
from typing import List
import numpy as np

from DenseLayer import DenseLayer
from support_functions import cross_entropy_loss, binary_cross_entropy_loss


class NeuralNetwork:

    def __check_dtype(self, layer):
        if not isinstance(layer, DenseLayer):
            raise ValueError("layers must be a list of DenseLayer objects")

    def __check_inputs_outputs(self, prev_layer_outputs, layer):
        if prev_layer_outputs is not None:
            if prev_layer_outputs != layer.weights.shape[0]:
                raise ValueError(
                    "Number of inputs must match outputs of previous layer")

    def __init__(self, layers: List[DenseLayer]):
        prev_layer_outputs = None
        for layer in layers:
            self.__check_dtype(layer)
            self.__check_inputs_outputs(prev_layer_outputs, layer)
            prev_layer_outputs = layer.weights.shape[1]
        self.layers = layers
        self.best_model = None

    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, target, outputs):
        loss = target - outputs
        for layer in reversed(self.layers):
            loss = layer.backward(loss)

    def training_step(self, input, target):
        outputs = self.feedforward(input)
        self.backpropagate(target, outputs)

    def shuffle(self, inputs, targets, rng):
        p = rng.permutation(len(inputs))
        return inputs[p], targets[p]

    def split(self, inputs, targets, percentage=0.2):
        i = int(len(inputs) // (1 / percentage))
        return inputs[i:], inputs[:i], targets[i:], targets[:i]

    # TODO: export model
    def fit(self, inputs, targets, epochs, plot_loss=False):
        train_learning_curve = []
        val_learning_curve = []
        rng = np.random.default_rng(42)

        inputs_train, inputs_val, targets_train, targets_val = self.split(
            inputs, targets)
        for epoch in range(epochs):
            inputs_train, targets_train = self.shuffle(
                inputs_train, targets_train, rng)

            for i, _ in enumerate(inputs_train):
                self.training_step(inputs_train[i], targets_train[i])

            outputs_train = np.array([self.feedforward(i)
                                     for i in inputs_train])
            train_loss = binary_cross_entropy_loss(
                targets_train, outputs_train)

            outputs_val = np.array([self.feedforward(i) for i in inputs_val])
            val_loss = binary_cross_entropy_loss(targets_val, outputs_val)
            print("Epoch %3d/%d - Training loss: %.2f - Val loss: %.2f" %
                  (epoch + 1, epochs, train_loss, val_loss))

            train_learning_curve.append(train_loss)
            val_learning_curve.append(val_loss)

        if plot_loss:
            plt.title("Cross entropy loss for training and validation")
            plt.plot(train_learning_curve, label='Training loss')
            plt.plot(val_learning_curve, label='Validation loss')
            plt.legend()
            plt.show()
