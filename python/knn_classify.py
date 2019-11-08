#!/usr/bin/python

import numpy
import operator

def create_data_set():
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify(inx, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diffMat = numpy.tile(inx, (data_set_size, 1))
    diffMat -= data_set
    sq_diffMat = diffMat**2
    sq_distances = sq_diffMat.sum(axis = 1)
    distances = sq_distances**0.5
    sorted_dist_indicies = distances.argsort()
    class_count={}
    for i in range(k):
        voteIlabel = labels[sorted_dist_indicies[i]]
        class_count[voteIlabel] = class_count.get(voteIlabel, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_class_count[0][0])

group, labels = create_data_set()
classify([0.1, 0], group, labels, 2)