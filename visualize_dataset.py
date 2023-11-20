import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cancer_data = pd.read_csv('breast_cancer_data.csv', index_col=0, header=None)
print(cancer_data.head())
# takes a while to run
sns.pairplot(data=cancer_data, hue=1)
plt.savefig('pairplot.png')

# good columns to use for classification, handpicked from pairplot
good_columns = [2, 4, 5, 7, 8, 9, 12, 14, 15, 22, 24, 25, 27, 28, 29]
