import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

eps = 1e-2
G = 100.


def newt(x, t):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = x
    pos = np.array([
        [x1, y1],
        [x2, y2],
        [x3, y3]
    ])
    r = np.stack((pos[:, :1] - pos[:, :1].T, pos[:, 1:2] - pos[:, 1:2].T), axis=2)
    d = np.sqrt(np.sum(r ** 2, axis=2, keepdims=True) + eps)
    dmask = (1. - np.eye(3))[..., None]

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


# x0 = [1., 0., 0., 1., -1., 0., 0., -1]
x0 = np.random.randn(12)

t = np.linspace(0, 1, 1001)

print("Simulating...")
sol = odeint(newt, x0, t)

print("Plotting...")
plt.figure()
plt.plot(sol[:, 0], sol[:, 1], 'b', sol[:, 4], sol[:, 5], 'r', sol[:, 8], sol[:, 9], 'g')
plt.savefig('trace.png')
plt.close()

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

print("Done.")
