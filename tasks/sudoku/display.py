import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec


def display(digit_logits: np.array, fname: str):
    """
    Renders a sudoku and saves it to a file

    :param digit_logits 81x10 probability of each position being a digit.
    :param fname: Where to save the rendered sudoku.
    """
    digit_probs = softmax(np.reshape(digit_logits[:, 1:], (9, 9, 9)), axis=2).reshape((9, 9, 3, 3))
    nrows = 1
    ncols = 9

    fig = plt.figure(figsize=(ncols, nrows))
    outer = gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0)

    digits = np.arange(1, 10).reshape(3, 3)
    for i in range(nrows):
        for j in range(ncols):
            rlw, clw = 1, 1

            if i % 3 == 0:
                rlw = 2
            if j % 3 == 0:
                clw = 2
            print(i, j)

            inner = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[i, j], wspace=0.0, hspace=0.0)
            for c in range(3):
                for r in range(3):
                    plt.subplot(inner[c, r])
                    plt.text(0.5, 0.5, str(digits[c, r]), horizontalalignment='center', verticalalignment='center', fontsize=10 * digit_probs[i, j, c, r] ** (1 / 4))
                    plt.xticks([])
                    plt.yticks([])

                    ax = fig.gca()
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    if ax.is_first_row():
                        top = ax.spines['top']
                        top.set_visible(True)
                        top.set_linewidth(rlw)
                    if ax.is_last_row():
                        bottom = ax.spines['bottom']
                        bottom.set_visible(True)
                        if i == nrows - 1 and (i + 1) % 3 == 0:
                            bottom.set_linewidth(2)
                    if ax.is_first_col():
                        left = ax.spines['left']
                        left.set_visible(True)
                        left.set_linewidth(clw)
                    if ax.is_last_col():
                        right = ax.spines['right']
                        right.set_visible(True)
                        if j == ncols - 1 and (j + 1) % 3 == 0:
                            right.set_linewidth(2)

    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()


def softmax(x, axis):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


# quiz: batch x 81
def quiz2logits(q):
    logits = -100 * np.ones((q.shape[0], 81, 10))
    for b in range(q.shape[0]):
        for d in range(q.shape[1]):
            if q[b, d] == 0:
                logits[b, d] = np.ones(10)
            else:
                for idx in map(int, list(str(q[b, d]))):
                    logits[b, d, idx] = 1

    return logits


if __name__ == '__main__':
    logits = -100 * np.ones((81, 10))
    logits[np.arange(81), np.random.randint(0, 10, size=(81,))] = 1
    logits[0, :] = -100
    display(logits, 'test.pdf')
