import numpy as np
from DenseLayer import DenseLayer
from NeuralNetwork import NeuralNetwork
from activation_functions import relu
from activation_functions import softmax

n = NeuralNetwork([
    DenseLayer(4, 8, relu),
    DenseLayer(8, 4, relu),
    DenseLayer(4, 2, softmax)
])

output = n.feedforward(np.array([1, 2, 3, 4]))
print(output)
