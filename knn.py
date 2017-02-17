import numpy as np
import operator as op


class KNN(object):

    def __init__(self):
        self.training_data = None
        self.training_labels = None
        self.labels = None
        pass

    def train(self, data, labels):
        # data is N x D where each row is a data point. labels is 1-dimension of size N
        # KNN classifier simply remembers all the training data
        self.training_data = data
        self.training_labels = labels
        self.labels = set(labels)

    def predict(self, data, k, l):
        y_predict = np.zeros(data.shape[0], dtype=self.training_labels.dtype)
        if l == 'L1':
            np.apply_along_axis(l1, 1, data, self.training_data, k, self.labels, self.training_labels, y_predict)
        else:
            l2(self.training_data, data, k, self.labels, self.training_labels, y_predict)
        return y_predict


def l1(b, a, k, names, ytr, y_pred):
    distances = np.sum(np.abs(a - b), axis=1)
    min_idx = np.argpartition(distances, k)
    # Get the k indexes corresponding to the lowest distances
    min_idx = min_idx[0:k]
    # Get the majority vote
    labels_count = dict((key, 0) for key in names)
    for j in range(k):
        labels_count[ytr[min_idx[j]]] += 1
    y_pred[l1.idx] = max(labels_count.items(), key=op.itemgetter(1))[0]
    l1.idx += 1
    if l1.idx == y_pred.shape[0]:
        l1.idx = 0
l1.idx = 0


def l2(a, b, k, names, ytr, y_pred):
    # (a + b)^2 = a^2 + b^2 - 2ab
    a_sum_square = np.sum(np.square(a), axis=1)
    b_sum_square = np.sum(np.square(b), axis=1)
    two_a_dot_bt = 2 * np.dot(a, b.T)
    # distances is a 2d array where each column is the distances of the respective testing data point
    distances = np.sqrt(a_sum_square[:, np.newaxis] + b_sum_square - two_a_dot_bt)
    for i in range(b.shape[0]):
        # Get ith column of distances and continue operations on it as normal (get lowest k)
        curr_distance = distances[:, i]
        min_idx = np.argpartition(curr_distance, k)
        # Get the k indexes corresponding to the lowest distances
        min_idx = min_idx[0:k]
        # Get the majority vote
        labels_count = dict((key, 0) for key in names)
        for j in range(k):
            labels_count[ytr[min_idx[j]]] += 1
        y_pred[i] = max(labels_count.items(), key=op.itemgetter(1))[0]
