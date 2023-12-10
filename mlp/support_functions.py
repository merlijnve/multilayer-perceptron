import numpy as np


def normalization(x):
    mean = np.mean(x, axis=len(x.shape) - 1, keepdims=True)
    std_dev = np.std(x, axis=len(x.shape) - 1, keepdims=True)

    return (x - mean) / std_dev


def binary_cross_entropy_loss(targets, outputs):
    return -np.mean(targets * np.log(outputs) + (1 - targets) *
                    np.log(1 - outputs))


def cross_entropy_loss(targets, outputs):
    return -np.mean(targets * np.log(outputs))


def to_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(filename, index_col=None):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            split = line.strip().split(',')
            if index_col is not None:
                split.pop(index_col)
            split = [to_float(value) for value in split]
            data.append(split)
    return data


def categorize_binary_data(data, y_col):
    types = dict()
    val = 0

    for index, row in enumerate(data):
        if row[y_col] not in types:
            types[row[y_col]] = val
            val += 1
        data[index][y_col] = types[row[y_col]]
    return data


def calc_opposite_class(y):
    return np.array([0 if element == 1 else 1 for element in y])


def read_cancer_dataset(path: str, categorize=True, index_col=None) -> np.ndarray:
    try:
        cancer_data = read_csv(path, index_col=index_col)
        if categorize:
            cancer_data = categorize_binary_data(cancer_data, 0)
        return np.array(cancer_data)
    except Exception as e:
        print(e)
        exit(1)
