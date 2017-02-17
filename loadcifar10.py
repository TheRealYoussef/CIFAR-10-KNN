import _pickle as pickle
import numpy as np
import os
import math


# Load single batch of cifar
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
    return datadict['data'].astype(np.float32), np.array(datadict['labels'])


# Load all of cifar
def load_cifar(folder):
    with open(os.path.join(folder, 'batches.meta'), 'rb') as f:
        names = pickle.load(f, encoding='latin1')
    training_data = np.empty([50000, 3072], dtype=np.float32)
    training_labels = np.empty([50000], dtype=np.int8)
    training_data_grayscale = np.empty([50000, 1024], dtype=np.float32)
    testing_data_grayscale = np.empty([10000, 1024], dtype=np.float32)
    for i in range(1, 6):
        start = (i - 1) * 10000
        end = i * 10000
        training_data[start:end], training_labels[start:end] = \
            load_cifar_batch(os.path.join(folder, 'data_batch_%d' % i))
    testing_data, testing_labels = load_cifar_batch(os.path.join(folder, 'test_batch'))
    for i in range(10000):
        for j in range(1024):
            training_data_grayscale[i, j] = float(math.floor((training_data[i, j] + training_data[i, j + 1024]
                                                              + training_data[i, j + 2048]) / 3))
            testing_data_grayscale[i, j] = float(math.floor((testing_data[i, j] + testing_data[i, j + 1024]
                                                             + testing_data[i, j + 2048]) / 3))
    for i in range(10000, 50000):
        for j in range(1024):
            training_data_grayscale[i, j] = math.floor((training_data[i, j]+training_data[i, j + 1024]
                                                        + training_data[i, j + 2048]) / 3)
    return training_data, training_data_grayscale, training_labels, testing_data, testing_data_grayscale,\
        testing_labels, names['label_names']


# Load part of cifar for cross validation
def load_cifar_cross_validation(folder, i):
    td = np.empty([4 * 10000, 3072], dtype=np.float32)
    tl = np.empty([4 * 10000], dtype=np.int8)
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
