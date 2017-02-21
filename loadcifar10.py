import _pickle as pickle
import numpy as np
import os
import math


# Load single batch of cifar
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
    return datadict['data'].astype(np.float64), np.array(datadict['labels'])


# Load all of cifar
def load_cifar(folder):
    with open(os.path.join(folder, 'batches.meta'), 'rb') as f:
        names = pickle.load(f, encoding='latin1')
    training_data = np.empty([50000, 3072], dtype=np.float64)
    training_labels = np.empty([50000], dtype=np.uint8)
    for i in range(1, 6):
        start = (i - 1) * 10000
        end = i * 10000
        training_data[start:end], training_labels[start:end] = \
            load_cifar_batch(os.path.join(folder, 'data_batch_%d' % i))
    testing_data, testing_labels = load_cifar_batch(os.path.join(folder, 'test_batch'))
    training_data_grayscale = training_data.reshape((50000, 3, 1024)).transpose((0, 2, 1))
    training_data_grayscale = np.mean(training_data_grayscale, axis=2)
    testing_data_grayscale = testing_data.reshape((10000, 3, 1024)).transpose((0, 2, 1))
    testing_data_grayscale = np.mean(testing_data_grayscale, axis=2)
    return training_data, training_data_grayscale, training_labels, testing_data, testing_data_grayscale,\
        testing_labels, names['label_names']


# Load part of cifar for cross validation
def load_cifar_cross_validation(folder, i):
    td = np.empty([40000, 3072], dtype=np.float64)
    tl = np.empty([40000], dtype=np.uint8)
    for j in range(1, 6):
        if i != j:
            if j < i:
                diff = 1
            else:
                diff = 2
            start = (j - diff) * 10000
            end = (j - diff + 1) * 10000
            td[start:end, :], tl[start:end] = \
                load_cifar_batch(os.path.join(folder, 'data_batch_%d' % j))
    vd, vl = load_cifar_batch(os.path.join(folder, 'data_batch_%d' % i))
    return td, tl, vd, vl
