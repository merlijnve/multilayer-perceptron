import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from support_functions import normalization


def save_pairplot(data, hue, filename):
    sns.pairplot(data=data, hue=hue)
    plt.savefig(filename)


cancer_data = pd.read_csv('breast_cancer_data.csv', index_col=0, header=None)
cancer_data = pd.get_dummies(cancer_data, dtype='float')

# takes a while to run because of the amount of features
save_pairplot(cancer_data, 1, 'pairplot.png')

cancer_data.iloc[:, :-2] = normalization(cancer_data.iloc[:, :-2])

# takes a while to run because of the amount of features
save_pairplot(cancer_data, '1_M', 'pairplot_normalized.png')
