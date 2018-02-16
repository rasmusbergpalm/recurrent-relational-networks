from itertools import permutations

import io
import matplotlib
from scipy.spatial.distance import cdist

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from PIL import Image


def switch_targets_paths(target_or_path):
    _, other = zip(*sorted([(j, i) for i, j in enumerate(target_or_path)]))
    return other


def greedy(cities: np.ndarray, n_jumps):
    left = cities.tolist()
    idx = list(range(len(cities)))
    path = [0]
    del left[0]
    del idx[0]
    for _ in range(n_jumps):
        dist = cdist(cities[path[-1]][None], np.array(left))
        closest = np.argmin(dist)
        path.append(idx[closest])
        del idx[closest]
        del left[closest]
    return path


def dist_squared(c1: np.ndarray, c2: np.ndarray):
    return np.sqrt(np.sum((c1 - c2) ** 2, axis=1))


class Greedy:
    def __init__(self, n):
        self.n = n

    def output_types(self):
        return tf.float32, tf.int32, tf.int32, tf.int32

    def output_shapes(self):
        return (self.n, 2), (self.n,), (self.n,), (self.n,)

    def sample_generator(self):
        while True:
            number_feature = np.arange(self.n)
            cities = np.random.uniform(size=(self.n, 2))
            path = greedy(cities, self.n - 1)
            targets = switch_targets_paths(path)

            yield cities, number_feature, targets, path


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
            path, min_length = self.brute(cities, self.n_fixed)
            targets = switch_targets_paths(path)

            yield cities, number_feature, targets, path


class PrettyClevr:
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray']

    def __init__(self, n):
        assert n <= len(self.colors), "Not enough colors defined."
        self.n = n
        self.r = 0.1

    def sample_generator(self):
        while True:
            object_colors = random.sample(range(self.n), self.n)
            objects = [np.random.uniform(size=(2,))]
            while len(objects) < self.n:
                o = np.random.uniform(size=(2,))
                if np.min(dist_squared(o, np.array(objects))) > self.r:
                    objects.append(o)

            n_jumps = np.random.randint(self.n)
            objects = np.array(objects)
            path = greedy(objects, n_jumps)
            target_obj = path[-1]
            anchor_color = object_colors[0]
            target_color = object_colors[target_obj]

            img = self._render(objects, object_colors)

            yield img, anchor_color, n_jumps, target_color

    def _render(self, objects, object_colors):
        fig = plt.figure(figsize=(1, 1), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        plt.xlim([0 - self.r, 1 + self.r])
        plt.ylim([0 - self.r, 1 + self.r])
        colors = [self.colors[c_idx] for c_idx in object_colors]
        ax.scatter(objects[:, 0], objects[:, 1], c=colors)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        im = Image.open(buf)

        return im.convert('RGB')


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
    d = PrettyClevr(8)
    gen = d.sample_generator()
    img, anchor_color, n_jumps, target_color = next(gen)

    fig = plt.figure(figsize=(1, 1), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img)
    name = "%s-%d-%s" % (d.colors[anchor_color], n_jumps, d.colors[target_color])
    plt.savefig(name)
    plt.close()

if __name__ == '__main2__':
    n = 7
    d = TSP(n)
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
