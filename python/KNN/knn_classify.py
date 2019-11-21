#!/usr/bin/python3

import numpy as np
from collections import Counter

class KNN_classifier:

	def __init__(self, k):
		assert k >= 1, "k must be integer"
		self.k = k
		self._X_train = None
		self._Y_train = None

	def fit(self, X_train, Y_train):
		assert X_train.shape[0] == Y_train.shape[0], \
			"the size of X_train must be equal to the size of Y_train."
		assert self.k <= X_train.shape[0], \
			"the size of X-train must be at least k."
		self._X_train = X_train
		self._Y_train = Y_train
		return self

	def predict_one(self, x):
		assert x.shape[0] == self._X_train.shape[1], \
			"the feature number of x must be equal to X_train"
		distances = []
		for xt in self._X_train:
			d = np.sum((xt - x)**2)
			distances.append(d)
		nearest = np.argsort(distances)
		topK_y = self._Y_train[nearest[:self.k]]
		votes = Counter(topK_y)
		return votes.most_common(1)[0][0]

	def predict(self, X_test):
		assert self._X_train is not None and self._Y_train is not None, \
			"must fit before predict!"
		assert X_test.shape[1] == self._X_train.shape[1], \
			"the feature number of X_predict must be equal to X_train"
		Y_predict = [self.predict_one(x) for x in X_test]
		return np.array(Y_predict)

	def score(self, Y_predict, Y_test):
		return sum(Y_predict == Y_test) / len(Y_test)