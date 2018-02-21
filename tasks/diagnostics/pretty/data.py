import io

import matplotlib

from tasks.diagnostics.data import greedy, dist_squared

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
from PIL import Image


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
            questions = []
            for start_idx in range(self.n):
                path = greedy(objects, self.n - 1, start_idx)
                for n_jumps in range(self.n):
                    for color_anchor in [True, False]:

                        target_obj = path[n_jumps]
                        if color_anchor:
                            anchor = self.colors[object_colors[start_idx]]
                            target = self.markers[object_markers[target_obj]]
                        else:
                            anchor = self.markers[object_markers[start_idx]]
                            target = self.colors[object_colors[target_obj]]

                        questions.append((self.s2i[anchor], n_jumps, self.s2i[target]))

            yield img, questions

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

    def generate(self, n):
        gen = self.sample_generator()
        with open('questions.csv', 'w') as qf:
            qf.write("# %s \n" % str(self.i2s))
            qf.write("# image, anchor, jumps, target\n")
            for i in range(n):
                img, questions = next(gen)
                png_name = '%05d.png' % i
                Image.fromarray(img).save(png_name)
                qf.writelines(["%s, %d, %d, %d\n" % (png_name, a, b, c) for (a, b, c) in questions])


if __name__ == '__main__':
    d = PrettyClevr(8)
    d.generate(10000)
