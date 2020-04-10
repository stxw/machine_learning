#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plot

def get_data():
	init_a = 2.2
	init_b = 4.5
	n_size = 1000
	x_train = np.random.random(size=n_size) * 20
	y_train = x_train * init_a + init_b + np.random.normal(scale=1, size=n_size)
	x_test = np.random.random(size=n_size) * 20
	y_test = x_test * init_a + init_b + np.random.normal(scale=1, size=n_size)
	return x_train, y_train, x_test, y_test

def cal_loss(y_predict, y_true):
	return np.sum(np.power(y_predict - y_true, 2)) / 2.0 / len(y_true)

def gradient_descent(x_train, y_train, eta=0.005, iter=100, batch_size=20, momentum=0.0):
	a = 0.0
	b = 0.0
	va = 0.0
	vb = 0.0
	al = list([a])
	bl = list([b])

	index = 0
	for i in range(iter):
		x_st = list()
		y_st = list()
		for j in range(index, index + batch_size):
			j = j % len(x_train)
			x_st.append(x_train[j])
			y_st.append(y_train[j])
		x_st = np.array(x_st)
		y_st = np.array(y_st)
		index = (index + batch_size) % len(x_train)

		y_predict = x_st * a + b
		loss = cal_loss(y_predict, y_st)
		grad_a = np.sum(x_st * (y_predict - y_st)) / batch_size
		grad_b = np.sum(1.0  * (y_predict - y_st)) / batch_size
		va = momentum * va - eta * grad_a
		vb = momentum * vb - eta * grad_b
		a = a + va
		b = b + vb
		al.append(a)
		bl.append(b)
	return a, b, al, bl

if __name__ == "__main__":
	x_train, y_train, x_test, y_test = get_data()
	a, b, al, bl = gradient_descent(x_train, y_train, momentum=0.9)
	print(a, b)
	# plot.scatter(x_train, y_train)
	# plot.plot(np.linspace(0, 20, 10), a * np.linspace(0, 20, 10) + b, '-r')
	# plot.show()
	plot.scatter(np.linspace(1, 100, 100), al, '-r')
	plot.show()