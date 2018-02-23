import glob
import io
import tarfile
import urllib.request
import zipfile

import matplotlib

from tasks.diagnostics.data import greedy, dist_squared

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import time


def fig2array(fig):
    with io.BytesIO() as buf:  # this is pretty stupid but it was the only way I could get it rendered with anti-aliasing
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf))


class PrettyClevr:
    url = "https://www.dropbox.com/s/3jaeq1ugcfs4jf3/pretty-clevr.tgz?dl=1"
    base_dir = (os.environ.get('DATA_DIR') or "/tmp")
    data_dir = base_dir + "/pretty-clevr"
    tgz_fname = base_dir + "/pretty-clevr.tgz"

    def __init__(self):
        if not os.path.exists(self.data_dir):
            print("Downloading data...")

            urllib.request.urlretrieve(self.url, self.tgz_fname)
            with tarfile.open(self.tgz_fname, "r:gz") as f:
                f.extractall(self.base_dir)

        with open(self.data_dir + '/questions.csv') as qf:
            print("Loading data...")
            with open(self.data_dir + '/dict.json') as fp:
                self.s2i = json.load(fp)
                self.i2s = {i: s for s, i in self.s2i.items()}

            self.questions = [l.strip().split(", ") for l in qf.readlines()]

        self.images = {img_fname: np.array(Image.open(img_fname).convert("RGB")) for img_fname in glob.glob(self.data_dir + '/images/*.png')}

    def output_types(self):
        return tf.uint8, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32

    def output_shapes(self):
        return (128, 128, 3), (8, 2), (8,), (), (), ()

    def sample_generator(self):
        while True:
            for img_fname, json_name, anchor, n_jumps, target in random.sample(self.questions, len(self.questions)):
                if n_jumps == "0":
                    with open(self.data_dir + '/states/' + json_name) as fp:
                        objects = json.load(fp)
                        positions = np.array([o['p'] for o in objects])
                        colors = [self.s2i[o['c']] for o in objects]

                    img = np.array(self.images[self.data_dir + '/images/' + img_fname], copy=True)

                    yield img, positions, colors, self.s2i[anchor], int(n_jumps), self.s2i[target]


class PrettyClevrGenerator:
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    markers = ['o', 'v', '^', '<', '>', 's', 'P', 'X']
    s2i = {v: i for i, v in enumerate(colors + markers)}
    i2s = {v: k for k, v in s2i.items()}
    n_input_output = len(s2i)

    def __init__(self, n):
        assert n <= len(self.colors), "Not enough colors defined."
        assert n <= len(self.markers), "Not enough markers defined."
        self.n = n
        self.r = 0.1

    def sample_generator(self):
        while True:
            objects = [{'p': p.tolist(), 'c': c, 'm': m} for p, c, m in zip(self.get_object_positions(), random.sample(self.colors, len(self.colors)), self.markers)]
            img = self._render(objects)
            questions = []
            for start_idx in range(self.n):
                path = greedy(np.array([o['p'] for o in objects]), self.n - 1, start_idx)
                for n_jumps in range(self.n):
                    for color_anchor in [True, False]:
                        target_obj = path[n_jumps]
                        if color_anchor:
                            anchor = objects[start_idx]['c']
                            target = objects[target_obj]['m']
                        else:
                            anchor = objects[start_idx]['m']
                            target = objects[target_obj]['c']

                        questions.append((anchor, n_jumps, target))

            yield objects, img, questions

    def get_object_positions(self):
        objects = [np.random.uniform(size=(2,))]
        while len(objects) < self.n:
            o = np.random.uniform(size=(2,))
            if np.min(dist_squared(o, np.array(objects))) > self.r:
                objects.append(o)
        return objects

    def _render(self, objects):
        fig = plt.figure(figsize=(1.28, 1.28), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.set_xlim(0 - self.r, 1 + self.r)
        ax.set_ylim(0 - self.r, 1 + self.r)
        for o in objects:
            x, y = o['p']
            ax.scatter(x, y, c=o['c'], marker=o['m'])

        return fig2array(fig)

    def generate(self, n):
        gen = self.sample_generator()
        with open('dict.json') as fp:
            json.dump(self.s2i, fp)

        with open('questions.csv', 'w') as qf:
            for i in range(n):
                objects, img, questions = next(gen)
                png_name = '%05d.png' % i
                json_name = '%05d.json' % i
                with open(json_name, 'w') as fp:
                    json.dump(objects, fp)
                Image.fromarray(img).save(png_name)
                qf.writelines(["%s, %s, %s, %d, %s\n" % (png_name, json_name, a, b, c) for (a, b, c) in questions])


if __name__ == '__main__':
    d = PrettyClevr()
    gen = d.sample_generator()
    start = time.perf_counter()
    for i in range(100000):
        next(gen)

    print((i + 1) / (time.perf_counter() - start), "Hz")
