#! /usr/bin/python3

import numpy as np

def split(X, Y, ratdio = 0.8):
	shuffle_index = np.random.permutation(len(X))
	train_size = int(len(X) * ratdio)
	test_size = len(X) - train_size
	train_indexes = shuffle_index[:train_size]
	test_indexes = shuffle_index[train_size:]

	X_train = X[train_indexes]
	X_test = X[test_indexes]
	Y_train = Y[train_indexes]
	Y_test = Y[test_indexes]
	return X_train, X_test, Y_train, Y_test

# X = np.linspace(1, 100, 100)
# Y = X
# X_train, X_test, Y_train, Y_test = split(X, Y, 0.8)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)