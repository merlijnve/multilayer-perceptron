import numpy as np
from numpy import ndarray


class Perceptron:
    def __init__(self, size):
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=size)
        self.bias = 1.0

    def estimate(self, inputs: ndarray):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match number of weights")

        sum = np.dot(inputs, self.weights) + self.bias
        # step/activation function
        output = np.sign(sum)
        return output

    def train(self, inputs: ndarray, target: ndarray):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match number of weights")

        est = self.estimate(inputs)
        error = (target - est)

        self.weights += error * inputs
        self.bias += error
