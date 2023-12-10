import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pickle
import copy
import time

from mlp.DenseLayer import DenseLayer
from mlp.support_functions import cross_entropy_loss
from mlp.NormalizationLayer import NormalizationLayer


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

    def __init__(self, layers: List[NormalizationLayer | DenseLayer],
                 early_stopping_n_epochs=10):
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
        self.early_stopping_n_epochs = early_stopping_n_epochs

        self.train_loss_curve = []
        self.train_accuracy_curve = []
        self.val_loss_curve = []
        self.val_accuracy_curve = []

    def predict(self, inputs):
        if isinstance(self.best_layers[0], NormalizationLayer):
            inputs = self.best_layers[0].transform(inputs)
        return self.feedforward(inputs, use_best=True)

    def feedforward(self, inputs, use_best=False):
        layers = self.layers if not use_best else self.best_layers
        for layer in layers:
            if isinstance(layer, DenseLayer):
                inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, loss):
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                loss = layer.backward(loss)

    def training_step(self, inputs, targets):
        outputs = self.feedforward(inputs)
        loss = targets - outputs
        self.backpropagate(loss)

    def shuffle(self, inputs, targets, rng):
        p = rng.permutation(len(inputs))
        return inputs[p], targets[p]

    def split(self, inputs, targets, percentage=0.2):
        i = int(len(inputs) // (1 / percentage))
        return inputs[i:], inputs[:i], targets[i:], targets[:i]

    def save_best_model(self, val_loss, epoch, significance=4):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        if self.best_val_loss is None or val_loss.round(significance) <= \
                self.best_val_loss.round(significance):
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_layers = copy.deepcopy(self.layers)

    def check_early_stopping(self, epoch):
        if epoch >= self.best_epoch + self.early_stopping_n_epochs:
            print("Early stopping...")
            return True
        return False

    def plot_curves(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        f.set_figwidth(12)

        ax1.set_title("Cross entropy loss")
        ax1.plot(self.train_loss_curve, label='Training loss')
        ax1.plot(self.val_loss_curve, label='Validation loss')
        ax1.set_xlabel("Epochs")
        ax1.legend()

        ax2.set_title("Accuracy")
        ax2.plot(self.train_accuracy_curve, label='Training accuracy')
        ax2.plot(self.val_accuracy_curve, label='Validation accuracy')
        ax2.set_ylim([0, 1])
        ax2.set_xlabel("Epochs")
        ax2.legend(loc='lower right')

        plt.show()

    def fit(self, inputs, targets, epochs, plotting=False, batch_size=None):
        t = time.time()
        rng = np.random.default_rng(42)

        inputs_train, inputs_val, targets_train, targets_val = self.split(
            inputs, targets)

        if isinstance(self.layers[0], NormalizationLayer):
            print("Fitting normalization layer")
            inputs_train = self.layers[0].fit_transform(inputs_train)
            inputs_val = self.layers[0].transform(inputs_val)

        if batch_size is not None and batch_size >= len(inputs_train):
            print("Ignoring batch size (bigger than train set)")
        for epoch in range(epochs):
            inputs_train, targets_train = self.shuffle(
                inputs_train, targets_train, rng)

            if batch_size is not None and batch_size < len(inputs_train):
                print("using batch")
                self.training_step(
                    inputs_train[:batch_size], targets_train[:batch_size])
            else:
                self.training_step(inputs_train, targets_train)

            outputs_train = self.feedforward(inputs_train)
            train_correct = outputs_train[:, 0].round(0) == targets_train[:, 0]
            train_loss = cross_entropy_loss(
                targets_train, outputs_train)

            outputs_val = self.feedforward(inputs_val)
            val_correct = outputs_val[:, 0].round(0) == targets_val[:, 0]
            val_loss = cross_entropy_loss(targets_val, outputs_val)
            self.save_best_model(val_loss, epoch)
            print("Epoch %3d/%d - Training loss: %.4f - Val loss: %.4f" %
                  (epoch + 1, epochs, train_loss, val_loss))

            self.train_loss_curve.append(train_loss)
            self.val_loss_curve.append(val_loss)
            self.val_accuracy_curve.append(sum(val_correct) / len(val_correct))
            self.train_accuracy_curve.append(
                sum(train_correct) / len(train_correct))
            if self.check_early_stopping(epoch):
                break

        print("Best epoch: %d - Val loss: %.4f" %
              (self.best_epoch, self.best_val_loss))
        print("Exporting model to file: best_model%.3f.pkl" %
              self.best_val_loss)
        pickle.dump(self, open("models/best_model%.3f.pkl" %
                    self.best_val_loss, "wb"))
        print("Finished in %.2f seconds" % (time.time() - t))
        if plotting:
            self.plot_curves()
