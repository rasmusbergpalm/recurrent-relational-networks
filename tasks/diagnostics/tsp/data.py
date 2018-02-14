from itertools import permutations
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import tensorflow as tf
import numpy as np


class TSP:
    n_fixed = 2

    def __init__(self, n):
        self.n = n

    def output_types(self):
        return tf.float32, tf.int32, tf.int32, tf.int32

    def output_shapes(self):
        return (self.n, 2), (self.n,), (self.n,), (self.n,)

    def brute(self, cities, n_fixed):
        def calc_length(cities, path):
            length = 0
            for i in range(len(path) - 1):
                length += dist_squared(cities[path[i]], cities[path[i + 1]])

            length += dist_squared(cities[path[-1]], cities[path[0]])
            return length

        def dist_squared(c1, c2):
            return np.sqrt(np.sum((c1 - c2) ** 2))

        min_length = float("inf")
        min_path = None

        for path in permutations(range(len(cities))[n_fixed:]):
            path = list(range(n_fixed)) + list(path)
            length = calc_length(cities, path)
            if length < min_length:
                min_length = length
                min_path = path

        return min_path, min_length

    def sample_generator(self):
        while True:
            number_feature = np.arange(self.n)
            cities = np.random.uniform(size=(self.n, 2))
            min_path, min_length = self.brute(cities, 2)
            targets = list({j: i for i, j in enumerate(min_path)}.values())

            yield cities, number_feature, targets, min_path


if __name__ == '__main__':
    d = TSP(9)
    start = time.perf_counter()
    gen = d.sample_generator()
    for i in range(10):
        c, f, t, p = next(gen)
    print((i + 1) / (time.perf_counter() - start), "Hz")

    print(p, t)
    plt.scatter(c[:, 0], c[:, 1])
    for i, txt in enumerate(range(d.n)):
        plt.annotate(txt, (c[i, 0], c[i, 1]))
    plt.plot(c[p, 0], c[p, 1])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('out.png')
