import numpy as np
from numpy import ndarray


class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.learning_rate = 0.1
        self.activation = activation

        np.random.seed(42)
        self.bias = np.zeros(output_size)
        self.weights = np.random.uniform(
            low=-1.0, high=1.0, size=(input_size, output_size))

    def forward(self, inputs: ndarray):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match number of weights")
        sum = self.activation(np.dot(inputs, self.weights) + self.bias)
        return sum
