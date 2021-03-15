#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import train_test_split
import knn_classify

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split.split(X, Y)
classifier = knn_classify.KNN_classifier(3)
classifier.fit(X_train, Y_train)
Y_predict = classifier.predict(X_test)
print(classifier.score(Y_predict, Y_test))