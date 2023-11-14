import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

from Perceptron import Perceptron
import numpy as np


def plot_iris(iris):
    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )
    plt.show()


def plot_learning_curve(learning_curve):
    _, ax = plt.subplots()
    ax.plot(learning_curve)
    ax.set(xlabel="Iterations", ylabel="Accuracy")
    plt.show()


def calc_accuracy(data, targets):
    correct = []
    for i, X in enumerate(data):
        inputs = np.array([X[0], X[1]])
        p.estimate(inputs)
        correct.append(p.estimate(inputs) == targets[i])

    return sum(correct) / len(correct)


def main():
    iris = load_iris()

    # reduce iris dataset to two classes
    iris.data = iris.data[iris.target != 2]
    iris.target = iris.target[iris.target != 2]

    data, targets = shuffle(iris.data, iris.target)

    # plot_iris(iris)

    global p
    p = Perceptron(size=2)

    targets = [-1.0 if t == 0 else 1.0 for t in targets]

    epochs = 150
    learning_curve = []
    for e in range(epochs):
        for i, X in enumerate(data):
            accuracies = []
            inputs = np.array([X[0], X[1]])
            p.train(inputs, targets[i])
            accuracies.append(calc_accuracy(data, targets))
        learning_curve.append(np.mean(accuracies))
        print("Epoch %4d" % e, "Accuracy: %.2f" % learning_curve[-1])

    plot_learning_curve(learning_curve)


if __name__ == "__main__":
    main()
