#! /usr/bin/python3

import numpy as np

class SimpleLinearRegression:

	def __init__(self):
		self.a_ = None
		self.b_ = None

	def fit(self, X_train, Y_train):
		assert X_train.ndim == 1, \
			"Simple Linear Regressor can only solve single feature training data."
		assert len(X_train) == len(Y_train), \
			"the size of Y_train must be equal to the size of Y_train"
		
		X_mean = np.mean(X_train)
		Y_mean = np.mean(Y_train)
		self.a_ = (X_train - X_mean).dot(Y_train - Y_mean) \
			/ (X_train - X_mean).dot(X_train - X_mean)
		self.b_ = Y_mean - self.a_ * X_mean
		return self

	def predict(self, X_test):
		assert X_test.ndim == 1, \
			"Simple Linear Regressor can only solve single feature training data."
		assert self.a_ is not None and self.b_ is not None, \
			"must fit before predict!"
		Y_predict = np.array([self.a_ * x + self.b_ for x in X_test])
		return Y_predict
	
	def score(self, Y_test, Y_predict):
		return 1 - np.sum((Y_predict - Y_test) ** 2) / len(Y_test) / np.var(Y_test)