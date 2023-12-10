import sys
import pickle

from mlp.support_functions import read_cancer_dataset
from mlp.support_functions import binary_cross_entropy_loss


def main():
    try:
        if len(sys.argv) != 3:
            print("""Usage: python3 predict.py <dataset filename>
                  <model filename>""")
            exit(1)

        nn = pickle.load(open(sys.argv[2], 'rb'))

        cancer_data_test = read_cancer_dataset(
            sys.argv[1], index_col=0)

        X = cancer_data_test[:, 1:]
        y = cancer_data_test[:, 0]

        preds = nn.predict(X)[:, 0]

        correct = preds.round(0) == y
        print("Correct: %d/%d" % (sum(correct), len(correct)))
        print("Accuracy: ", sum(correct) / len(correct))

        loss = binary_cross_entropy_loss(y, preds)
        print("Binary cross entropy loss: ", loss)

    except Exception as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
