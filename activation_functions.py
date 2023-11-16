import numpy as np


class Activation:
    def function(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid():
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sigmoid_x = self.function(x)
        return sigmoid_x * (1 - sigmoid_x)


class ReLU():
    def function(self, x):
        return np.maximum(0.0, x)

    def derivative(self, x):
        return np.heaviside(x, 1.0)


class Softmax():
    def function(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    # TODO: check if this is correct
    def derivative(self, x):
        softmax_val = self.function(x)
        return softmax_val * (1 - softmax_val)
