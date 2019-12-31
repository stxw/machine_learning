#!/usr/bin/python3

import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
print("\n\n", len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print("\n", feature.shape, feature.dtype)
print(label, type(label), label.dtype)
x, y = mnist_train[0:9]
d2l.show_fashion_mnist(x, d2l.get_fashion_mnist_labels(y))

batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith("win"):
	num_work = 0
else:
	num_work = 4
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
	batch_size, shuffle=True, num_workers=num_work)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
	batch_size, shuffle=True, num_workers=num_work)

start = time.time()
for x, y in train_iter:
	continue
print((time.time() - start), "sec")