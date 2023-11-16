import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def function(self, x):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, sigmoid_x):
        return sigmoid_x * (1 - sigmoid_x)


class ReLU(Activation):
    def function(self, x):
        return np.maximum(0.0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class Softmax(Activation):
    def function(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def derivative(self, x):
        softmax_val = self.function(x)
        return softmax_val * (1 - softmax_val)
