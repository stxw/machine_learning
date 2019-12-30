#!/usr/bin/python3

from mxnet import nd, autograd
import numpy as np
from matplotlib import pyplot as plt
import random

num_inputs = 2
num_examples = 30
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# plt.figure(num=1)
# plt.scatter(features[:, 0].asnumpy(), labels.asnumpy(), 1)
# plt.figure(num=2)
# plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
# plt.show()

def data_iter(features, labels, batch_size):
	num_examples = len(features)
	indices = list(range(num_examples))
	random.shuffle(indices)
	for i in range(0, num_examples, batch_size):
		sub_indices = nd.array(indices[i: min(i + batch_size, num_examples)])
		yield features.take(sub_indices), labels.take(sub_indices)

# for x, y in data_iter(features, labels, 10):
# 	print(x, y)

w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()

print(nd.dot(features, w) + b)