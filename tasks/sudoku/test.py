import os

import matplotlib
import numpy as np

from tasks.sudoku.display import quiz2logits, display
from tasks.sudoku.rrn import SudokuRecurrentRelationalNet

matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_dir = '/nobackup/titans/rapal/models/sudoku/2c9bfa6/'
SudokuRecurrentRelationalNet.n_steps = n_steps = 64
render_steps = True

eval_fname = model_dir + '%d-eval.npz' % n_steps
if not os.path.exists(eval_fname):
    model = SudokuRecurrentRelationalNet(True)

    model.load(model_dir + "best")
    quizzes, logits, answers, i = [], [], [], 0
    try:
        while True:
            quiz, logit, answer = model.test_batch()
            quizzes.append(quiz)
            logits.append(logit)
            answers.append(answer)
            i += 1
            print(i)
    except Exception as e:  # It should throw a tf.errors.OutOfRangeError, but sometimes it throws another exception, so we'll catch em all
        np.savez(eval_fname, logits=logits, quizzes=quizzes, answers=answers)
        print(e)

data = np.load(eval_fname)
quizzes = data['quizzes'].reshape(-1, 81)
logits = data['logits']  # n_batches, n_steps, batch_size, 81, 10
answers = data['answers'].reshape(-1, 81)

puzzle_correct = 0
chars_correct = 0
acc_fname = model_dir + '%d-acc.npz' % n_steps
if not os.path.exists(acc_fname):
    print("net")
    acc = np.zeros((n_steps, 18, 2))
    for step in range(n_steps):
        step_predictions = np.argmax(logits[:, step, ...], axis=-1).reshape(-1, 81)
        eq = np.equal(step_predictions, answers)
        n_givens = np.sum(quizzes != 0, axis=1) - 17
        for i in range(18):
            idx = n_givens == i
            digit_acc = np.mean(eq[idx])
            puzzle_acc = np.mean(np.all(eq[idx], axis=1))
            acc[step, i] = [digit_acc, puzzle_acc]

    np.savez(acc_fname, acc=acc)

data = np.load(acc_fname)
acc = data['acc']


def heatmap(results, idx, name):
    plt.figure(figsize=(8, 8))
    plt.imshow(results[:, :, idx].T, cmap='viridis', origin='lower', interpolation='none', vmin=0.0, vmax=1.0)
    plt.ylabel('extra givens')
    plt.xlabel('steps')
    plt.title(name)
    plt.colorbar()
    plt.savefig(model_dir + name)
    plt.close()


heatmap(acc, 0, '%d-digit-acc.pdf' % n_steps)
heatmap(acc, 1, '%d-puzzle-acc.pdf' % n_steps)

plt.figure(figsize=(8, 4))
for i in range(0, 18, 2):
    plt.plot(np.arange(n_steps) + 1, acc[:, i, 1], label='%d givens' % (i + 17))

plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Steps')
plt.axvline(x=32, ls='--', c='black')
plt.grid()
plt.savefig(model_dir + '%d-graph-puzzle-acc.pdf' % n_steps)
plt.close()

for i in range(18):
    print("%d givens: %f acc" % (i + 17, acc[-1, i, 1]))

print("Rendering steps... This will take a long time.")
if render_steps:
    n_givens = np.sum(quizzes != 0, axis=1)
    idx17 = n_givens == 17

    N = 10
    quizzes = quizzes[idx17][:N]  # (N, 81)
    logits0 = quiz2logits(quizzes)  # (N, 81, 10)
    logits = np.transpose(logits, (0, 2, 1, 3, 4))  # (n_batches, n_steps, batch_size, 81, 10)
    logits = logits.reshape(-1, n_steps, 81, 10)  # (n_batch*batch_size, n_steps, 81, 10)
    logits = logits[idx17][:N]  # (N, n_steps, 81, 10)
    logits = np.concatenate([logits0[:, np.newaxis, ...], logits], axis=1)  # (N, n_steps+1, 81, 10)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            display(logits[i, j], model_dir + "%03d-%02d.pdf" % (i, j))
