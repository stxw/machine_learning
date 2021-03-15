#! /usr/bin/python3

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets
import train_test_split
import time

digits = datasets.load_digits()
X = digits.data
Y = digits.target
X_train, X_test, Y_train, Y_test = train_test_split.split(X, Y)
knn_clf = KNeighborsClassifier()

param_grid = [
	{
		"weights" : ["uniform"],
		"n_neighbors" : [i for i in range(1, 11)]
	},
	{
		"weights" : ["distance"],
		"n_neighbors" : [i for i in range(1, 11)],
		"p" : [i for i in range(1, 6)]
	}
]

start_t = time.perf_counter()
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)
grid_search.fit(X_train, Y_train)
end_t = time.perf_counter()
print(grid_search.best_score_)
print(grid_search.best_params_)
print("time cost", end_t - start_t, "s")