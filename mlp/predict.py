import pickle
from mlp.support_functions import read_cancer_dataset
from mlp.support_functions import binary_cross_entropy_loss

nn = pickle.load(open('models/best_model.pkl', 'rb'))

cancer_data_test = read_cancer_dataset(
    "data/breast_cancer_data.csv_test", index_col=0)

X = cancer_data_test[:, 1:]
y = cancer_data_test[:, 0]

preds = nn.predict(X)[:, 0]

correct = preds.round(0) == y
print("Correct: %d/%d" % (sum(correct), len(correct)))
print("Accuracy: ", sum(correct) / len(correct))


loss = binary_cross_entropy_loss(y, preds)
print("Binary cross entropy loss: ", loss)
# TODO: check requirements
