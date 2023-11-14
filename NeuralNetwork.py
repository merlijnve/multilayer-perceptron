from DenseLayer import DenseLayer
from typing import List


class NeuralNetwork:

    def __check_dtype(self, layer):
        if not isinstance(layer, DenseLayer):
            raise ValueError("layers must be a list of DenseLayer objects")

    def __check_inputs_outputs(self, prev_layer_outputs, layer):
        if prev_layer_outputs is not None:
            if prev_layer_outputs != layer.weights.shape[0]:
                raise ValueError(
                    "Number of inputs must match outputs of previous layer")

    def __init__(self, layers: List[DenseLayer]):
        prev_layer_outputs = None
        for l in layers:
            self.__check_dtype(l)
            self.__check_inputs_outputs(prev_layer_outputs, l)
            prev_layer_outputs = l.weights.shape[1]

        self.layers = layers

    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
