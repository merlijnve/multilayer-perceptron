import numpy as np
from DenseLayer import DenseLayer
from NeuralNetwork import NeuralNetwork
from activation_functions import Softmax, Sigmoid

n = NeuralNetwork([
    DenseLayer(2, 2, Sigmoid()),
    DenseLayer(2, 2, Sigmoid())
])

X = np.array([[0.0, 0.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])


y = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0]
])


n.train(X, y, 100)

print("GUESSES")
print(n.feedforward([0.0, 0.0]))
print(n.feedforward([1.0, 0.0]))
print(n.feedforward([0.0, 1.0]))
print(n.feedforward([1.0, 1.0]))

