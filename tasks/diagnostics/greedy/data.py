import matplotlib
from scipy.spatial.distance import cdist

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform


class Greedy:
    def __init__(self, n):
        self.n = n

    def output_types(self):
        return tf.float32, tf.int32, tf.int32, tf.int32

    def output_shapes(self):
        return (self.n, 2), (self.n,), (self.n,), (self.n,)

    def greedy(self, cities):
        left = cities.tolist()
        idx = list(range(len(cities)))
        path = [0]
        del left[0]
        del idx[0]
        while len(left) > 0:
            dist = cdist(cities[path[-1]][None], np.array(left))
            closest = np.argmin(dist)
            path.append(idx[closest])
            del idx[closest]
            del left[closest]
        return path

    def sample_generator(self):
        while True:
            number_feature = np.arange(self.n)
            cities = np.random.uniform(size=(self.n, 2))
            path = self.greedy(cities)
            targets = list({j: i for i, j in enumerate(path)}.values())

            yield cities, number_feature, targets, path


def plot_path(cities, path):
    plt.scatter(cities[:, 0], cities[:, 1])
    for i, txt in enumerate(range(d.n)):
        plt.annotate(txt, (cities[i, 0], cities[i, 1]))
    plt.plot(cities[path, 0], cities[path, 1])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('out.png')
    plt.close()


if __name__ == '__main__':
    n = 5
    d = Greedy(n)
    start = time.perf_counter()
    gen = d.sample_generator()
    for i in range(10):
        c, f, t, p = next(gen)
    print((i + 1) / (time.perf_counter() - start), "Hz")

    print(p, t)
    plot_path(c, p)
    X = squareform(pdist(c))
    d0 = zip(X[0, :].tolist(), range(n))
    print([i for d, i in sorted(d0, key=lambda x: x[0])])
    plt.imshow(X)
    plt.savefig('dist.png')
    plt.close()
