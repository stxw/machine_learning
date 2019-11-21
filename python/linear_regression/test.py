#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plot

plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x - 40) ** 2 - 80
# plot.plot(plot_x, plot_y)
# plot.show()

def dJ(theta):
	return 2 * (theta - 2.5)

def J(theta):
	return (theta - 2.5) ** 2 - 1

def gradient_descent(initial_theta=0.0, eta=0.1, n_step=10000, epsilon=1e-8):
	theta = initial_theta
	for i in range(1, n_step):
		gradient = dJ(theta)
		last_theta = theta
		theta = theta - eta * gradient
		if(abs(gradient) < epsilon):
			print("break", i)
			break
	return theta

theta = gradient_descent()
print(theta)
print(J(theta))