from DenseLayer import DenseLayer
from NeuralNetwork import NeuralNetwork
from activation_functions import Softmax, Sigmoid, ReLU
from support_functions import normalization
import pandas as pd

cancer_data = pd.read_csv('breast_cancer_data.csv', index_col=0, header=None)

# categorize the diagnosis column
cancer_data = pd.get_dummies(cancer_data, dtype='float')

# TODO: make preprocessing a layer
# apply z-score normalization to feature columns
cancer_data.iloc[:, :-2] = normalization(cancer_data.iloc[:, :-2])
print(cancer_data.head())


X = cancer_data.iloc[:, :12].to_numpy()
y = cancer_data.iloc[:, -2:].to_numpy()

n = NeuralNetwork([
    DenseLayer(X.shape[1], 16, Sigmoid()),
    DenseLayer(16, 16, Sigmoid()),
    DenseLayer(16, 2, Softmax())
])

n.fit(X, y, epochs=300, plot_loss=True)

correct = []
for i in range(len(X)):
    pred = n.feedforward(X[i])
    correct.append(pred.round(0)[0] == y[i][0])
print("Accuracy: ", sum(correct) / len(X))
