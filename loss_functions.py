import numpy as np


def binary_cross_entropy_loss(targets, outputs):
    return -np.sum(targets * np.log(outputs) + (1 - targets) *
                   np.log(1 - outputs))


def cross_entropy_loss(targets, outputs):
    return -np.sum(targets * np.log(outputs))
