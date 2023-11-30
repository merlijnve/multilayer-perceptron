from mlp.support_functions import normalization
import numpy as np

x = np.array([0.7972,  0.0767,  0.4383,  0.7866,  0.8091,
              0.1954,  0.6307,  0.6599,  0.1065,  0.0508])
expected = np.array([1.1273, -1.247, -0.0552, 1.0923, 1.1664,
                     -0.8559, 0.5786, 0.6748, -1.1488, -1.3324])
x_normal = normalization(x)
print("Correct normalization values: ",
      np.array_equal(np.round(x_normal, decimals=3),
                     np.round(expected, decimals=3)))


x = np.array([[5, 6, 7, 7, 8],
              [8, 8, 8, 9, 9],
              [2, 2, 4, 4, 5]])
expected = np.array([[-1.569, -0.588, 0.392, 0.392, 1.373],
                     [-0.816, -0.816, -0.816, 1.225, 1.225],
                     [-1.167, -1.167, 0.5, 0.5, 1.333]])

x_normal = normalization(x)
print("Works on multiple dimension arrays: ",
      np.array_equal(np.round(x_normal, decimals=3),
                     np.round(expected, decimals=3)))
