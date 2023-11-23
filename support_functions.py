import numpy as np
import pandas as pd


def normalization(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    mean = np.mean(x, axis=len(x.shape) - 1, keepdims=True)
    std_dev = np.std(x, axis=len(x.shape) - 1, keepdims=True)

    x_normal = (x - mean) / std_dev
    return pd.DataFrame(x_normal)


def binary_cross_entropy_loss(targets, outputs):
    return -np.sum(targets * np.log(outputs) + (1 - targets) *
                   np.log(1 - outputs)) / len(targets)


def cross_entropy_loss(targets, outputs):
    return -np.sum(targets * np.log(outputs)) / len(targets)
