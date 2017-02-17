import knn
import loadcifar10 as lc
import numpy as np
import threading


def apply_knn(k, l, td, tl, ted, tel, ln, s):
    knn_o = knn.KNN()
    knn_o.train(td, tl)
    pred = knn_o.predict(ted, k, l)
    count = dict((key, 0) for key in ln)
    for i in pred:
        count[ln[i]] += 1
    for k, v in count.items():
        print('%s %s accuracy = %f' % (k, s, v / 1000))
    print('%s accuracy = %f' % (s.capitalize(), np.sum(pred == tel) / 10000))


def main():
    td, tdg, tl, ted, tedg, tel, ln = lc.load_cifar('cifar-10-batches-py')
    k = 1
    l = 'L2'
    colored = threading.Thread(target=apply_knn, args=(k, l, td, tl, ted, tel, ln, 'colored'))
    grayscale = threading.Thread(target=apply_knn, args=(k, l, tdg, tl, tedg, tel, ln, 'grayscale'))
    colored.start()
    grayscale.start()
    colored.join()
    grayscale.join()

if __name__ == '__main__':
    main()
