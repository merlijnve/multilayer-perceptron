import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
