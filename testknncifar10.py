import knn
import loadcifar10 as lc
import numpy as np
import threading
import time
import matplotlib.pyplot as plt

lock = threading.Lock()
results = dict()
root = 'cifar-10-batches-py'


def cross_validate(l, k):
    accuracy = np.empty([5], dtype=np.float32)
    for i in range(1, 6):
        time_start = time.time()
        td, tl, vd, vl = lc.load_cifar_cross_validation(root, i)
        knn_o = knn.KNN()
        knn_o.train(td, tl)
        predictions = knn_o.predict(vd, k, l)
        num_correct = np.sum(predictions == vl)
        accuracy[i - 1] = num_correct / 10000
        print(time.time() - time_start)
    with lock:
        results[k] = accuracy


def plot_data(l, res):
    for k, v in sorted(res.items()):
        plt.scatter([k] * len(v), v)
    # Plot the trend line with error bars that correspond to standard deviation
    mean = np.array([np.mean(v) for k, v in sorted(res.items())])
    std = np.array([np.std(v) for k, v in sorted(res.items())])
    plt.errorbar([k for k, v in sorted(res.items())], mean, yerr=std)
    plt.title('Cross-validation on %s and k' % (l, ))
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


def test_cross_validate(l):
    t = [None] * 4
    ks = [1, 3, 5, 7, 9, 10, 13, 17, 20, 50, 75, 100]
    for i in range(len(ks) // 4):
        for j in range(4):
            t[j] = threading.Thread(target=cross_validate, args=(l, ks[4 * i + j]))
            t[j].start()
        for j in range(4):
            t[j].join()
    plot_data(l, results)
    results.clear()


def main():
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    test_cross_validate('L2')
    test_cross_validate('L1')

if __name__ == '__main__':
    main()
