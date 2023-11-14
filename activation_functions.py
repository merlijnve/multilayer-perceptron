import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return np.heaviside(x, 1.0)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))
