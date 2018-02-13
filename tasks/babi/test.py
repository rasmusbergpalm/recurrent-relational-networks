import os
import numpy as np

from tasks.babi.rrn import BaBiRecurrentRelationalNet

model_dir = '/home/rapal/runs/527c7c5/'
n_steps = BaBiRecurrentRelationalNet.n_steps

eval_fname = model_dir + '%d-eval.npz' % n_steps
if not os.path.exists(eval_fname):
    model = BaBiRecurrentRelationalNet(True)

    model.load(model_dir + "best")
    batches = model.test_batches()
    np.savez(eval_fname, batches=batches)

data = np.load(eval_fname)
batches = data['batches']

logits = np.concatenate(batches[:, 0], axis=1)  # (5, 40k, 177)
answers = np.concatenate(batches[:, 1], axis=0)  # (40k, )
task_indices = np.concatenate(batches[:, 2], axis=0)  # (40k, )

for i in range(20):
    idx = task_indices == i
    expected = answers[idx]
    actual = np.argmax(logits[-1, idx, :], axis=1)
    acc = np.mean(expected == actual)
    print(acc)
