import io
import os
import matplotlib

from tasks.diagnostics.data import greedy, dist_squared

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
from PIL import Image
import time


def fig2array(fig):
    with io.BytesIO() as buf:  # this is pretty stupid but it was the only way I could get it rendered with anti-aliasing
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf)
        return np.array(im.convert('RGB'))


class PrettyClevr:
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

    def output_types(self):
        return tf.uint8, tf.float32, tf.int32, tf.int32, tf.int32

    def output_shapes(self):
        return (128, 128, 3), (128, 128, 2), (), (), ()

    def sample_generator(self):
        while True:
            object_colors = random.sample(range(self.n), self.n)
            object_markers = random.sample(range(self.n), self.n)
            objects = self.get_objects()
            img = self._render(objects, object_colors, object_markers)
            xy = np.stack(np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128)), axis=-1)
            path = greedy(objects, self.n - 1)
            for n_jumps in range(self.n):
                for color_anchor in [True, False]:

                    target_obj = path[n_jumps]
                    if color_anchor:
                        anchor = self.colors[object_colors[0]]
                        target = self.markers[object_markers[target_obj]]
                    else:
                        anchor = self.markers[object_markers[0]]
                        target = self.colors[object_colors[target_obj]]

                    yield img, xy, self.s2i[anchor], n_jumps, self.s2i[target]

    def get_objects(self):
        objects = [np.random.uniform(size=(2,))]
        while len(objects) < self.n:
            o = np.random.uniform(size=(2,))
            if np.min(dist_squared(o, np.array(objects))) > self.r:
                objects.append(o)
        objects = np.array(objects)
        return objects

    def _render(self, objects, object_colors, object_markers):
        fig = plt.figure(figsize=(1.28, 1.28), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.set_xlim(0 - self.r, 1 + self.r)
        ax.set_ylim(0 - self.r, 1 + self.r)
        colors = [self.colors[c_idx] for c_idx in object_colors]
        markers = [self.markers[m_idx] for m_idx in object_markers]
        for (x, y), c, m in zip(objects, colors, markers):
            ax.scatter(x, y, c=c, marker=m)

        return fig2array(fig)


if __name__ == '__main__':
    # TODO pre-generate dataset. For speed and ensure there's no fishy matplotlib thread-safety issues.
    data_dir = os.environ.get('DATA_DIR') or '/tmp'
    d = PrettyClevr(8)
    gen = d.sample_generator()
    n = 16 * 10000
    start = time.perf_counter()

    for i in range(n):
        img, xy, anchor, n_jumps, target = next(gen)
        fig = plt.figure(figsize=(2.56, 2.56), frameon=False)
        plt.imshow(img)
        plt.title('foo')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        name = "%s-%d-%s" % (d.i2s[anchor], n_jumps, d.i2s[target])
        plt.savefig(name)
        plt.close()
