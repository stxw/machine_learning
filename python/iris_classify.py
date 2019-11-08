#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
Y = iris.target

shuffle_indexes = np.random.permutation(len(X))
test_ratio = 0.2
test_size = len(X) * test_ratio
test_size = (int)(test_size)
test_indexes = shuffle_indexes[:test_size]
train_indexes = shuffle_indexes[test_size:]
x_train = X[train_indexes]
y_train = Y[train_indexes]
y_test = Y[test_indexes]
x_test = X[test_indexes]

for i in range(X.shape[0]):
    print(X[i][0], X[i][1], X[i][2], X[i][3], Y[i])