import numpy as np

from NeuralNetwork import NeuralNetwork
from DenseLayer import DenseLayer
from NormalizationLayer import NormalizationLayer
from activation_functions import Softmax, Sigmoid
from support_functions import read_cancer_dataset, calc_opposite_class


def main():
    cancer_data = read_cancer_dataset()

    X = cancer_data[:, 1:]
    y = np.column_stack(
        [cancer_data[:, 0], calc_opposite_class(cancer_data[:, 0])])

    n = NeuralNetwork([
        NormalizationLayer(),
        DenseLayer(X.shape[1], 64, Sigmoid(), learning_rate=0.01),
        DenseLayer(64, 64, Sigmoid(), learning_rate=0.01),
        DenseLayer(64, 2, Softmax(), learning_rate=0.01)
    ])

    n.fit(X, y, epochs=200, plot_loss=True)

    predictions = n.predict(X)

    correct = predictions[:, 0].round(0) == y[:, 0]
    print("Train accuracy: ", sum(correct) / len(X))


if __name__ == '__main__':
    main()
