import numpy as np


def normalization(x: np.ndarray):
    mean = np.mean(x)
    std_dev = np.std(x)

    x_normal = (x - mean) / std_dev
    return x_normal


def binary_cross_entropy_loss(targets, outputs):
    return -np.sum(targets * np.log(outputs) + (1 - targets) *
                   np.log(1 - outputs))


def cross_entropy_loss(targets, outputs):
    return -np.sum(targets * np.log(outputs))
