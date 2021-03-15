#!/usr/bin/python3

import numpy as np
from sklearn import datasets
import train_test_split
import knn_classify

digits = datasets.load_digits()
X = digits.data
Y = digits.target
# print(X.shape)
# print(Y.shape)

# import matplotlib.pyplot as plt
# some_digit = X[666]
# some_digit_image = some_digit.reshape(8, 8)
# plt.imshow(some_digit_image)
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split.split(X, Y)
classifier = knn_classify.KNN_classifier(3)
classifier.fit(X_train, Y_train)
Y_predict = classifier.predict(X_test)
print(classifier.score(Y_predict, Y_test))