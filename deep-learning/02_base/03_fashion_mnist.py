#!/usr/bin/python3

import sys
import time
import numpy as np
from mxnet.gluon import data as g_data
import d2lzh as d2l

mnist_train = g_data.vision.FashionMNIST(train=True)
mnist_test = g_data.vision.FashionMNIST(train=False)
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, feature.dtype)

def get_fashion_mnist_labels(labels):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', \
		'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]
print(get_fashion_mnist_labels([label]))

def show_fashion_mnist(images, labels):
	d2l.use_svg_display()
	_, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
	for f, img, lbl in zip(figs, images, labels):
		f.imshow(img.reshape((28, 28)).asnumpy())
		f.set_title(lbl)
		f.axes.get_xaxis().set_visible(False)
		f.axes.get_yaxis().set_visible(False)
x, y = mnist_test[0:9]
show_fashion_mnist(x, get_fashion_mnist_labels(y))