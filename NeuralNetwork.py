import matplotlib.pyplot as plt
from typing import List

from DenseLayer import DenseLayer
from loss_functions import cross_entropy_loss


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

    # TODO: add validation set
    # TODO: add early stopping
    # TODO: add minibatch training
    def fit(self, inputs, targets, epochs=100, plot_loss=False):
        learning_curve = []

        for epoch in range(epochs):
            for i, _ in enumerate(inputs):
                self.training_step(inputs[i], targets[i])

            outputs = [self.feedforward(i) for i in inputs]
            loss = cross_entropy_loss(targets, outputs)
            print("Epoch %3d/%d - Training loss: %.2f" %
                  (epoch + 1, epochs, loss))
            learning_curve.append(loss)

        if plot_loss:
            plt.plot(learning_curve)
            plt.show()
