import numpy as np
from numpy import ndarray


class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.output = None
        self.activation = activation

        self.learning_rate = 0.1

        np.random.seed(42)
        self.bias = np.zeros(output_size)
        self.weights = np.random.uniform(
            low=-1.0, high=1.0, size=(input_size, output_size))

    def forward(self, inputs: ndarray):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match number of weights")
        self.inputs = inputs
        self.output = self.activation.function(
            np.dot(inputs, self.weights) + self.bias)
        return self.output

    def backward(self, loss_output):
        output_gradient = loss_output * self.activation.derivative(self.output)
        # print("output gradient", output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= self.learning_rate * \
            np.outer(self.inputs.T, output_gradient)
        self.bias -= self.learning_rate * output_gradient

        # print("W", np.sum(self.weights))
        # print("B", np.sum(self.bias))
        return input_gradient
