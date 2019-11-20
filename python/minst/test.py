#! /usr/bin/python

import network

size = [2, 3, 1]
ntk = network.Network(size)
print(ntk.biases)
print(ntk.weights)
print(size[1:])
print(size[:-1])