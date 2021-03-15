#! /usr/bin/python3

import numpy as np

class Simple_ann:

	def __init__(self, size):
		self.size = size
		self.weight_ = [np.random.randn(next_size, previous_size) \
			for previous_size, next_size in zip(size[:-1], size[1:])]

	# sigmoid transformation
	def sigmoid(self, input):
		return 1.0 / (1.0 + np.exp(-input))

	# forword pass
	def forword_pass(self, X_train):
		X = X_train
		out = []
		out.append(X)
		for i in range(len(self.size) - 1):
			Y = self.weight_[i].dot(X)
			Y = [self.sigmoid(y) for y in Y]
			X = np.array(Y)
			out.append(X)
		return X, out

	# back pass
	def back_pass(self, Y_true, Y, out, eta=0.1):
		new_w = [np.zeros([next_size, previous_size]) \
			for previous_size, next_size in zip(self.size[:-1], self.size[1:])]
		m = np.shape(np.array(Y))[1]

		cur_loss_out = np.array(Y - Y_true)
		cur_loss_net = cur_loss_out * out[-1] * (1 - out[-1])
		for k in range(len(self.size) - 1):
			k = len(self.size) - 2 - k
			gradient = cur_loss_net.dot(out[k].T) / m
			new_w[k] = self.weight_[k] - eta * gradient
			cur_loss_out = self.weight_[k].T.dot(cur_loss_net)
			cur_loss_net = cur_loss_out * out[k] * (1 - out[k])
		self.weight_ = new_w

	def train(self, X_train, Y_train, iter_time=100000, eta=0.1):
		X_train = np.array(X_train).T
		Y_train = np.array(Y_train).T
		for i in range(iter_time):
			Y, out = self.forword_pass(X_train)
			self.back_pass(Y_train, Y, out, eta)
		return self

	# predict
	def predict(self, X_predict):
		X = np.array(X_predict).T
		for i in range(len(self.size) - 1):
			Y = self.weight_[i].dot(X)
			Y = [self.sigmoid(y) for y in Y]
			X = np.array(Y)
		Y_predict = X.T
		return Y_predict

	def get_loss(self, Y_predict, Y_true):
		Y_true = np.array(Y_true).T
		return np.sum((Y_predict - Y_true) ** 2) / Y_true.shape[1]
