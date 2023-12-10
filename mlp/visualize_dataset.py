import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mlp.support_functions import normalization


def save_pairplot(data, hue, filename):
    print("Making pairplot...")
    sns.pairplot(data=data, hue=hue)
    plt.savefig(filename)
    print("Done")


def main():
    try:
        cancer_data = pd.read_csv(
            'data/breast_cancer_data.csv', index_col=0, header=None)
        cancer_data = pd.get_dummies(cancer_data, dtype='float')
        print(cancer_data.head())

        # takes a while to run because of the amount of features
        save_pairplot(cancer_data, '1_M', 'images/pairplot.png')

        print("Normalizing...")
        cancer_data.iloc[:, :-2] = pd.DataFrame(
            normalization(cancer_data.iloc[:, :-2].to_numpy()))

        # takes a while to run because of the amount of features
        save_pairplot(cancer_data, '1_M', 'images/pairplot_normalized.png')

    except Exception as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
