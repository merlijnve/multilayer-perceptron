import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mlp.support_functions import normalization


def save_pairplot(data, hue, filename):
    sns.pairplot(data=data, hue=hue)
    plt.savefig(filename)


cancer_data = pd.read_csv(
    'data/breast_cancer_data.csv', index_col=0, header=None)
cancer_data = pd.get_dummies(cancer_data, dtype='float')
print(cancer_data.head())

# takes a while to run because of the amount of features
print("Making pairplot...")
save_pairplot(cancer_data, '1_M', 'images/pairplot.png')

cancer_data.iloc[:, :-2] = normalization(cancer_data.iloc[:, :-2])

# takes a while to run because of the amount of features
print("Making normalized pairplot...")
save_pairplot(cancer_data, '1_M', 'images/pairplot_normalized.png')


# TODO: fix this
# Traceback (most recent call last):
#   File "<frozen runpy>", line 198, in _run_module_as_main
#   File "<frozen runpy>", line 88, in _run_code
#   File "/Users/merlijn/Projects/Codam/multilayer-perceptron/mlp/visualize_dataset.py", line 21, in <module>
#     cancer_data.iloc[:, :-2] = normalization(cancer_data.iloc[:, :-2])
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/merlijn/Projects/Codam/multilayer-perceptron/mlp/support_functions.py", line 5, in normalization
#     mean = np.mean(x, axis=len(x.shape) - 1, keepdims=True)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.11/site-packages/numpy/core/fromnumeric.py", line 3502, in mean
#     return mean(axis=axis, dtype=dtype, out=out, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.11/site-packages/pandas/core/generic.py", line 11556, in mean
#     return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.11/site-packages/pandas/core/generic.py", line 11201, in mean
#     return self._stat_function(
#            ^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.11/site-packages/pandas/core/generic.py", line 11154, in _stat_function
#     nv.validate_stat_func((), kwargs, fname=name)
#   File "/usr/local/lib/python3.11/site-packages/pandas/compat/numpy/function.py", line 80, in __call__
#     validate_kwargs(fname, kwargs, self.defaults)
#   File "/usr/local/lib/python3.11/site-packages/pandas/util/_validators.py", line 163, in validate_kwargs
#     _check_for_default_values(fname, kwds, compat_args)
#   File "/usr/local/lib/python3.11/site-packages/pandas/util/_validators.py", line 79, in _check_for_default_values
#     raise ValueError(
# ValueError: the 'keepdims' parameter is not supported in the pandas implementation of mean()
