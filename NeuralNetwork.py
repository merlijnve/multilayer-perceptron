import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pickle
import copy

from DenseLayer import DenseLayer
from support_functions import binary_cross_entropy_loss
from NormalizationLayer import NormalizationLayer


class NeuralNetwork:

    def __check_dtype(self, layer):
        if not isinstance(layer, DenseLayer):
            raise ValueError("layers must be a list of DenseLayer objects")

    def __check_inputs_outputs(self, prev_layer_outputs, layer):
        if prev_layer_outputs is not None:
            if prev_layer_outputs != layer.weights.shape[0]:
                raise ValueError(
                    "Number of inputs must match outputs of previous layer")

    def __check_preprocessing_layer(self, layer, index):
        if isinstance(layer, NormalizationLayer) and index != 0:
            raise ValueError("Preprocessing layer must be first layer")

    def __init__(self, layers: List[NormalizationLayer | DenseLayer]):
        prev_layer_outputs = None
        for index, layer in enumerate(layers):
            if isinstance(layer, NormalizationLayer):
                self.__check_preprocessing_layer(layer, index)
            if isinstance(layer, DenseLayer):
                self.__check_dtype(layer)
                self.__check_inputs_outputs(prev_layer_outputs, layer)
                prev_layer_outputs = layer.weights.shape[1]
        self.layers = layers
        self.best_val_loss = None
        self.best_epoch = 0

    def predict(self, inputs):
        if isinstance(self.best_layers[0], NormalizationLayer):
            inputs = self.best_layers[0].transform(inputs)
        return np.array([self.feedforward(i, use_best=True) for i in inputs])

    def feedforward(self, inputs, use_best=False):
        layers = self.layers if not use_best else self.best_layers
        for layer in layers:
            if isinstance(layer, DenseLayer):
                inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, target, outputs):
        loss = target - outputs
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                loss = layer.backward(loss)

    def training_step(self, input, target):
        outputs = self.feedforward(input)
        self.backpropagate(target, outputs)

    def shuffle(self, inputs, targets, rng):
        p = rng.permutation(len(inputs))
        return inputs[p], targets[p]

    def split(self, inputs, targets, percentage=0.2):
        i = int(len(inputs) // (1 / percentage))
        return inputs[i:], inputs[:i], targets[i:], targets[:i]

    def save_best_model(self, val_loss, epoch):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_layers = copy.deepcopy(self.layers)

    def check_early_stopping(self, epoch, n_epochs=20):
        if epoch >= self.best_epoch + n_epochs:
            print("Early stopping...")
            return True
        return False

    def fit(self, inputs, targets, epochs, plot_loss=False):
        train_learning_curve = []
        val_learning_curve = []
        rng = np.random.default_rng(42)

        inputs_train, inputs_val, targets_train, targets_val = self.split(
            inputs, targets)

        if isinstance(self.layers[0], NormalizationLayer):
            print("Fitting normalization layer")
            inputs_train = self.layers[0].fit_transform(inputs_train)
            inputs_val = self.layers[0].transform(inputs_val)

        for epoch in range(epochs):
            inputs_train, targets_train = self.shuffle(
                inputs_train, targets_train, rng)

            for i, _ in enumerate(inputs_train):
                self.training_step(inputs_train[i], targets_train[i])

            outputs_train = np.array([self.feedforward(i)
                                     for i in inputs_train])
            train_loss = binary_cross_entropy_loss(
                targets_train, outputs_train)

            outputs_val = np.array([self.feedforward(i) for i in inputs_val])
            val_loss = binary_cross_entropy_loss(targets_val, outputs_val)
            self.save_best_model(val_loss, epoch)
            print("Epoch %3d/%d - Training loss: %.2f - Val loss: %.2f" %
                  (epoch + 1, epochs, train_loss, val_loss))

            train_learning_curve.append(train_loss)
            val_learning_curve.append(val_loss)
            if self.check_early_stopping(epoch, n_epochs=20):
                break

        print("Best epoch: %d - Val loss: %.4f" %
              (self.best_epoch, self.best_val_loss))
        pickle.dump(self, open("best_model.pkl", "wb"))
        if plot_loss:
            plt.title("Cross entropy loss for training and validation")
            plt.plot(train_learning_curve, label='Training loss')
            plt.plot(val_learning_curve, label='Validation loss')
            plt.legend()
            plt.show()
