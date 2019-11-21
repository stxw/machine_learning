#! /usr/bin/python3

import numpy as np

class StandardScaler:

	def __init__(self):
		self.mean_ = None
		self.scale_ = None

	def fit(self, X):
		assert X.ndim == 2, "The dimension of X must be 2"
		self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
		self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
		return self

	def transform(self, X):
		assert X.ndim == 2, "The dimension of X must be 2"
		assert self.mean_ is not None and self.scale_ is not None, \
			"must fit before transform!"
		resX = np.empty(shape=X.shape, dtype=float)
		for i in range(X.shape[1]):
			resX[:, i] = (X[:, i] - self.mean_[i]) / self.scale_[i]
		return resX