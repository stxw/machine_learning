#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import train_test_split
import knn_classify

# 最值归一化
x1 = np.random.randint(0, 100, [50, 2])
x1 = np.array(x1, dtype=float)
x1[:, 0] = (x1[:, 0] - np.min(x1[:, 0])) / (np.max(x1[:, 0]) - np.min(x1[:, 0]))
x1[:, 1] = (x1[:, 1] - np.min(x1[:, 1])) / (np.max(x1[:, 1]) - np.min(x1[:, 1]))
print(np.mean(x1), np.std(x1))
# plt.scatter(x1[:, 0], x1[:, 1])
# plt.show()

# 均值归一化
x2 = np.random.randint(0, 100, [50, 2])
x2 = np.array(x2, dtype=float)
x2[:, 0] = (x2[:, 0] - np.mean(x2[:, 0])) / np.std(x2[:, 0])
x2[:, 1] = (x2[:, 1] - np.mean(x2[:, 1])) / np.std(x2[:, 1])
print(np.mean(x2), np.std(x2))
# plt.scatter(x2[:, 0], x2[:, 1])
# plt.show()


iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split.split(X, Y)

# 使用sklearn的StandardScaler
# from sklearn.preprocessing import StandardScaler

# 使用自己写的StandarScaler
from standard_scaler import StandardScaler

std_scaler = StandardScaler()
std_scaler.fit(X_train)
print(std_scaler.mean_)
print(std_scaler.scale_)
X_train = std_scaler.transform(X_train)
X_test = std_scaler.transform(X_test)

# X_train_mean = []
# X_train_std = []
# for i in range(X_train.shape[1]):
#     X_train_mean.append(np.mean(X_train[:, i]))
#     X_train_std.append(np.std(X_train[:, i]))
# print(X_train_mean)
# print(X_train_std)
# for i in range(X_train.shape[1]):
#     X_train[:, i] = (X_train[:, i] - X_train_mean[i]) / X_train_std[i]
#     X_test[:, i] = (X_test[:, i] - X_train_mean[i]) / X_train_std[i]

classifier = knn_classify.KNN_classifier(3)
classifier.fit(X_train, Y_train)
Y_predict = classifier.predict(X_test)
print(classifier.score(Y_predict, Y_test))