import numpy as np
from mlp.DenseLayer import DenseLayer
from mlp.NeuralNetwork import NeuralNetwork
from mlp.activation_functions import Softmax, Sigmoid

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


def train_nn():
    n.fit(X, y, 50, plot_loss=False)

    print("GUESSES")
    # Test the trained network
    for inputs in X:
        prediction = n.feedforward(inputs)
        print(f"Input: {inputs}, Prediction: {prediction.round(2)}")


def test_shuffle():
    rng = np.random.default_rng(42)
    inputs, targets = n.shuffle(X, y, rng)
    print(inputs, targets)


def test_split():
    inputs_train, inputs_val, targets_train, targets_val = n.split(X, y, 0.2)
    print("inputs train", inputs_train, "\ninputs val",
          inputs_val, "\ntargets train", targets_train, "\ntargets val",
          targets_val)


# test_shuffle()
test_split()
