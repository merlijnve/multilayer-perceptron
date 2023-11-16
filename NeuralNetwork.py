from DenseLayer import DenseLayer
from typing import List
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(4)


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

    def binary_cross_entropy_loss(self, targets, outputs):
        return -np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))

    # TODO: move this to a loss function class
    def cross_entropy_loss(self, targets, outputs):
        return -np.sum(targets * np.log(outputs))

    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, targets, outputs):
        # print("targets", targets)
        # print("outputs", outputs)
        loss = targets - outputs
        print(targets, outputs, loss)
        for layer in reversed(self.layers):
            loss = layer.backward(loss)

    def train(self, inputs, targets, epochs):
        learning_curve = []
        for epoch in range(epochs):
            print("Epoch %4d" % epoch)
            # TODO: shuffle inputs and targets
            i = random.randint(0, len(inputs) - 1)
            # for i, _ in enumerate(inputs):
            print(i)
            outputs = self.feedforward(inputs[i])
            self.backpropagate(targets[i], outputs)
            cross_entropy_loss = self.binary_cross_entropy_loss(targets[i], outputs)
            print("Loss %.2f" % cross_entropy_loss)
            learning_curve.append(cross_entropy_loss)

        plt.plot(learning_curve)
        plt.show()
