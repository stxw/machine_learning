#! /usr/bin/python3

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plot
import linear_regression

boston = datasets.load_boston()
X = boston.data
Y = boston.target
X = X[Y < 50]
Y = Y[Y < 50]
import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split.split(X, Y)

reg = linear_regression.LinearRegression()
reg.fit_normal(X_train, Y_train)
Y_predict = reg.predict(X_test)
print(reg.theta_)
print(reg.score(Y_test, Y_predict))