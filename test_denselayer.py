from DenseLayer import DenseLayer
import numpy as np
from activation_functions import relu

layer = DenseLayer(4, 4, relu)
input = np.array([1, 2, 3, 4])
output = layer.forward(input)
print(output)
output = layer.forward(output)
print(output)
output = layer.forward(output)
print(output)
