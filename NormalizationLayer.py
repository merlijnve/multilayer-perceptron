import numpy as np
import pandas as pd


class NormalizationLayer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
