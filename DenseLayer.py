import numpy as np
from numpy import ndarray


class DenseLayer:
    def __init__(self, input_size, output_size, activation,
                 learning_rate=0.01):
        self.output = None
        self.activation = activation
        self.learning_rate = learning_rate

        self.bias = np.zeros(output_size)
        self.weights = np.random.default_rng(42).uniform(
            low=-1.0, high=1.0, size=(input_size, output_size))

    def forward(self, inputs: ndarray):
        self.input = inputs
        self.output = self.activation.function(
            np.dot(inputs, self.weights) + self.bias)
        return self.output

    def backward(self, loss_output):
        output_gradient = loss_output * self.activation.derivative(self.output)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights += self.learning_rate * \
            np.dot(self.input.T, output_gradient)
        self.bias += self.learning_rate * np.sum(output_gradient, axis=0)

        return input_gradient
