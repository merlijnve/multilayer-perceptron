from support_functions import normalization
import numpy as np

x = np.array([1, 2, 3, 4])
x_normal = normalization(x)
print("Correct normalization values: ",
      np.array_equal(np.round(x_normal, decimals=2), np.array([-1.34, -0.45, 0.45, 1.34])))

x = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
x_normal = normalization(x)
print("Works on multiple dimension arrays: ", x_normal.shape == (2, 4))
