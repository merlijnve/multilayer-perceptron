import numpy as np
from DenseLayer import DenseLayer
from NeuralNetwork import NeuralNetwork
from activation_functions import Softmax, Sigmoid


n = NeuralNetwork([
    DenseLayer(2, 256, Sigmoid()),
    DenseLayer(256, 256, Sigmoid()),
    DenseLayer(256, 2, Softmax())
])

X = np.array([[0.0, 0.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])


y = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
])


n.fit(X, y, 50, plot_loss=True)

print("GUESSES")
# Test the trained network
for inputs in X:
    prediction = n.feedforward(inputs)
    print(f"Input: {inputs}, Prediction: {prediction.round(2)}")
