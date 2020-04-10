#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plot

def get_data():
	init_a = 1.0
	init_b = 0.5
	n_size = 1000
	x_train = np.random.random(size=n_size) * 5
	y_train = x_train * init_a + init_b + np.random.normal(scale=0.2, size=n_size)
	return x_train, y_train

def cal_loss(y_predict, y_true):
	return np.sum(np.power(y_predict - y_true, 2)) / 2.0 / len(y_true)

def gradient_descent(x_train, y_train, eta=0.005, iter=200, batch_size=1000, momentum=0.0):
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
	return a, b, np.array(al), np.array(bl)

if __name__ == "__main__":
	x_train, y_train = get_data()
	style = ['pink', '-b', '-g', '-y']

	plot.figure("a")
	for i in range(4):
		plot.ylim(0, 1.7)
		mom = float(i) / 10.0 * 3
		a, b, al, bl = gradient_descent(x_train, y_train, eta=0.005, iter=100, momentum=mom)
		print("%.1f" % (mom), ":", a, b)
		plot.plot(np.linspace(0, len(al), len(al)), al, style[i], label = "%.1f" % (mom))
	plot.plot(np.linspace(0, len(al), 2), np.array([1, 1]), '-r', label = "dst")
	plot.legend()
	
	plot.figure("b")
	for i in range(4):
		plot.ylim(0, 0.6)
		mom = float(i) / 10.0 * 3
		a, b, al, bl = gradient_descent(x_train, y_train, eta=0.005, iter=200, momentum=mom)
		print("%.1f" % (mom), ":", a, b)
		plot.plot(np.linspace(0, len(bl), len(bl)), bl, style[i], label = "%.1f" % (mom))
	plot.plot(np.linspace(0, len(al), 2), np.array([0.5, 0.5]), '-r', label = "dst")
	plot.legend()
	plot.show()