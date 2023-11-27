from NeuralNetwork import NeuralNetwork
from DenseLayer import DenseLayer
from NormalizationLayer import NormalizationLayer
from activation_functions import Softmax, Sigmoid
import pandas as pd

cancer_data = pd.read_csv('breast_cancer_data.csv', index_col=0, header=None)

# categorize the diagnosis column
cancer_data = pd.get_dummies(cancer_data, dtype='float')

X = cancer_data.iloc[:, :12].to_numpy()
y = cancer_data.iloc[:, -2:].to_numpy()

n = NeuralNetwork([
    NormalizationLayer(),
    DenseLayer(X.shape[1], 16, Sigmoid()),
    DenseLayer(16, 16, Sigmoid()),
    DenseLayer(16, 2, Softmax())
])

n.fit(X, y, epochs=70, plot_loss=False)

predictions = n.predict(X)

correct = predictions[:, 0].round(0) == y[:, 0]
print("Accuracy: ", sum(correct) / len(X))
