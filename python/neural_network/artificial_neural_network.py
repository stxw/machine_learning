#! /usr/bin/python3

import numpy as np

Xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
# the input of the operation

Ys  = np.array([[0, 0, 0, 1],	# the answer of and operation
				[0, 1, 1, 1],	# the answer of or operation
				[0, 1, 1, 0],	# the answer of xor operation
				[1, 1, 1, 0]	# the answer of nand operation
				])

size = [2, 5, 8, 8, 4]
# the size of neural network
# the number "2" means there are two input variables
# the number "5, 6" means there are two hidden neurons
# the number "4" means there is one output of this neural network

weight = [np.random.randn(next_size, previous_size) for previous_size, next_size in zip(size[:-1], size[1:])]
# initialize the weight in random


# sigmoid transformation
def sigmoid(input):
	return 1.0 / (1.0 + np.exp(-input))


# forword pass
def forword_pass(X_train):
	X = np.array(X_train)
	out = []
	out.append(X)

	for i in range(len(size) - 1):
		Y = weight[i].dot(X)
		Y = [sigmoid(y) for y in Y]
		X = np.array(Y)

		out.append(X)
	return X, out


# back pass
def back_pass(Y_true, Y, out, eta = 0.1):
	new_w = weight
	m = np.shape(np.array(Y))[1]

	cur_loss_out = np.array(Y - Y_true)
	cur_loss_net = cur_loss_out * out[-1] * (1 - out[-1])
	for k in range(len(size) - 1):
		k = len(size) - 2 - k
		gradient = cur_loss_net.dot(out[k].T) / m
		new_w[k] = new_w[k] - eta * gradient
		cur_loss_out = weight[k].T.dot(cur_loss_net)
		cur_loss_net = cur_loss_out * out[k] * (1 - out[k])


# predict
def predict(X):
	X = np.array(X)
	for i in range(len(size) - 1):
		Y = weight[i].dot(X)
		Y = [sigmoid(y) for y in Y]
		X = np.array(Y)
	return X


if __name__ == "__main__":
	for i in range(50000):
		Y, out = forword_pass(Xs)
		back_pass(Ys, Y, out, eta=1.5)
	Y_predict = predict(Xs)
	print(Y_predict)