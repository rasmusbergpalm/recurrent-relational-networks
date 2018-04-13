import io
import urllib.request

import matplotlib
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import time
import progressbar
import os

eps = 1e-2
G = 100.
dmask = (1. - np.eye(3))[..., None]


def newt(x, t):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = x
    pos = np.array([
        [x1, y1],
        [x2, y2],
        [x3, y3]
    ])
    r = np.stack((pos[:, :1] - pos[:, :1].T, pos[:, 1:2] - pos[:, 1:2].T), axis=2)
    d = np.sqrt(np.sum(r ** 2, axis=2, keepdims=True) + eps)
    a = np.sum(dmask * G * (1. / d ** 3) * r, axis=0)  # (3, 2)

    dxdt = [
        vx1,
        vy1,
        a[0, 0],
        a[0, 1],
        vx2,
        vy2,
        a[1, 0],
        a[1, 1],
        vx3,
        vy3,
        a[2, 0],
        a[2, 1],
    ]
    return dxdt


class NBody:
    url = "https://www.dropbox.com/s/93943xmp4pvu1f2/3body.npz?dl=1"
    base_dir = (os.environ.get('DATA_DIR') or "/tmp")
    data_fname = base_dir + "/3body.npz"

    def __init__(self):
        if not os.path.exists(self.data_fname):
            print("Downloading data...")
            urllib.request.urlretrieve(self.url, self.data_fname)

        with np.load(self.data_fname) as data:
            samples = data['samples']
            self.dev = self.sampler(samples[:10000])
            self.train = self.sampler(samples[10000:])

    def sampler(self, samples):
        while True:
            np.random.shuffle(samples)
            for s in samples:
                yield s

    def sample_generator(self, ):
        t = np.linspace(0, 1, 1024)
        while True:
            x0 = np.random.randn(12)
            sol = odeint(newt, x0, t)
            sol = sol.reshape(1024, 3, 4)
            for a in np.split(sol, 8, axis=0):
                yield np.transpose(a, (1, 0, 2)).astype(np.float32)

    def trace_diff(self, e, a):
        plt.figure()
        for i, c in zip(range(3), ['b-', 'r-', 'g-']):
            plt.plot(e[:, i, 0], e[:, i, 1], c, alpha=0.5)
        for i, c in zip(range(3), ['b--', 'r--', 'g--']):
            plt.plot(a[:, i, 0], a[:, i, 1], c, alpha=0.5)

        with io.BytesIO() as buf:
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return np.array(Image.open(buf))

    def trace(self, sol):
        plt.figure()
        plt.plot(sol[:, 0], sol[:, 1], 'b', sol[:, 4], sol[:, 5], 'r', sol[:, 8], sol[:, 9], 'g')
        plt.savefig('trace.png')
        plt.close()

    def animate(self, sol):
        print("Animating...")
        fig = plt.figure(figsize=(8, 8))
        pos = sol[:, (0, 1, 4, 5, 8, 9)]

        ax = fig.add_subplot(111, autoscale_on=False, xlim=(pos.min(), pos.max()), ylim=(pos.min(), pos.max()))
        ax.grid()
        p1, = ax.plot([], [], 'bo-', lw=2)
        p2, = ax.plot([], [], 'ro-', lw=2)
        p3, = ax.plot([], [], 'go-', lw=2)

        def anim(sol):
            p1.set_data(sol[0], sol[1])
            p2.set_data(sol[4], sol[5])
            p3.set_data(sol[8], sol[9])
            return p1, p2, p3

        ani = FuncAnimation(fig, anim, sol, blit=True)
        ani.save('trace.mp4', fps=24)


if __name__ == '__main__':
    start = time.perf_counter()
    nbody = NBody()
    g = nbody.sample_generator()
    samples = []
    stop = 100000
    for i in progressbar.progressbar(range(stop)):
        samples.append(next(g))
    np.savez_compressed('samples.npz', samples=samples)
