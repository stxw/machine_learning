#! /usr/bin/python3

from simple_ann import Simple_ann
import numpy as np

Xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
# the input of the operation

Ys = [	[0, 0, 0, 1],
		[0, 1, 1, 0],
		[0, 1, 1, 1],
		[1, 1, 0, 0]]

size = [2, 5, 4]
# the size of neural network

sann = Simple_ann(size)
sann.train(Xs, Ys, iter_time=5000, eta=10.0)
Y_predict = sann.predict(Xs)
print(Y_predict)
loss = sann.get_loss(Y_predict, Ys)
print(loss)