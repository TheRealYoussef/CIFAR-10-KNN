import numpy as np
import scipy.spatial.distance as sp


class KNN(object):

    def __init__(self):
        pass

    def train(self, data, labels):
        # data is N x D where each row is a data point. labels is 1-dimension of size N
        # KNN classifier simply remembers all the training data
        self.training_data = data
        self.training_labels = labels

    def predict(self, data, k, l):
        # data is M x D where each row is a data point, k is the number of nearest neighbours, l  is the distance metric ('L1' or 'L2')
        # y_predict is the predicted labels of data
        y_predict = np.zeros(data.shape[0], dtype=self.training_labels.dtype)
        if l == 'L1':
            self.l1(data, k, y_predict)
        else:
            self.l2(data, k, y_predict)
        return y_predict

    def l1(self, data, k, y_pred):
        # data is M x D where each row is a data point, k is the number of nearest neighbours, y_pred is the predicted labels of data
        # Compute Manhattan distance, distances is M x N where each row 'i' is the distances of the ith data point from the training data points
        distances = sp.cdist(data, self.training_data, 'cityblock')
        for i in range(data.shape[0]):
            # Get ith row of distances
            curr_distance = distances[i]
            # Get the k indexes corresponding to the lowest distances
            min_idx = np.argpartition(curr_distance, k)[0:k]
            # Get the votes
            votes = self.training_labels[min_idx]
            # Count the votes
            labels_count = np.bincount(votes)
            # Choose the majority vote
            y_pred[i] = np.argmax(labels_count)

    def l2(self, data, k, y_pred):
        # data is M x D where each row is a data point, k is the number of nearest neighbours, y_pred is the predicted labels of data
        # (a + b)^2 = a^2 + b^2 - 2ab
        a_sum_square = np.sum(np.square(self.training_data), axis=1)
        b_sum_square = np.sum(np.square(data), axis=1)
        two_a_dot_bt = 2 * np.dot(self.training_data, data.T)
        # Compute Euclidean distance, distances is N x M where each column 'i' is the distances of the ith data point from the training data points
        distances = np.sqrt(a_sum_square[:, np.newaxis] + b_sum_square - two_a_dot_bt)
        for i in range(data.shape[0]):
            # Get ith column of distances and continue operations on it as normal (get lowest k)
            curr_distance = distances[:, i]
            # Get the k indexes corresponding to the lowest distances
            min_idx = np.argpartition(curr_distance, k)[0:k]
            # Get the votes
            votes = self.training_labels[min_idx]
            # Count the votes
            labels_count = np.bincount(votes)
            # Choose the majority vote
            y_pred[i] = np.argmax(labels_count)
