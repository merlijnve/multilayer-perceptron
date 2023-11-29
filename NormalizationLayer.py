import numpy as np


class NormalizationLayer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, x):
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
