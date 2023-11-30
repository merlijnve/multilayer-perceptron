import numpy as np

from mlp.NeuralNetwork import NeuralNetwork
from mlp.DenseLayer import DenseLayer
from mlp.NormalizationLayer import NormalizationLayer
from mlp.activation_functions import Softmax, Sigmoid
from mlp.support_functions import read_cancer_dataset, calc_opposite_class


def main():
    cancer_data = read_cancer_dataset()

    X = cancer_data[:, 1:]
    y = np.column_stack(
        [cancer_data[:, 0], calc_opposite_class(cancer_data[:, 0])])

    n = NeuralNetwork([
        NormalizationLayer(),
        DenseLayer(X.shape[1], 64, Sigmoid(), learning_rate=0.01),
        DenseLayer(64, 64, Sigmoid(), learning_rate=0.01),
        DenseLayer(64, 2, Softmax(), learning_rate=0.1)
    ], early_stopping_n_epochs=20)

    n.fit(X, y, epochs=400, plot_loss=True, batch_size=None)

    predictions = n.predict(X)

    correct = predictions[:, 0].round(0) == y[:, 0]
    print("Correct: %d/%d" % (sum(correct), len(correct)))
    print("Accuracy: ", sum(correct) / len(correct))


if __name__ == '__main__':
    main()
