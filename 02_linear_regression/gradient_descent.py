#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plot

np.random.seed(666)
X = 2 * np.random.random(size=100)
Y = 3.0 * X + 4 + np.random.normal(size=100)
X = X.reshape(-1, 1)

plot.scatter(X, Y)
plot.show()

def cost(theta, X_b, Y):
	return np.sum((Y - X_b.dot(theta)) ** 2) / len(X_b)

def dcost(theta, X_b, Y):
	return X_b.T.dot(X_b.dot(theta) - y) * 2.0 / len(X_b)

def 