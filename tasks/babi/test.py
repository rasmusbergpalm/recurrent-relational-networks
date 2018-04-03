import glob
import os

import numpy as np
import requests
from os.path import relpath

from tasks.babi.rrn import BaBiRecurrentRelationalNet

test_revisions = ['4e1d56c', 'ac440ad', '889fcb4', '7c0c35e', 'e144ca9', 'a58ea1a', '5273506', 'dd91e9a']
model_dir = '/home/rapal/runs'
tensorboard_dir = os.environ.get('TENSORBOARD_DIR') + '/bAbI/debug/'
tensorboard_url = "http://localhost:6007/data/plugin/scalars/scalars"


def extract_scalars(run, tag):
    response = requests.get(tensorboard_url, params={"run": run, "tag": tag, "format": "json"})
    return response.json()


def get_run_name(revision):
    runs = glob.glob(tensorboard_dir + revision + '/test/' + revision + '*')
    assert len(runs) == 1, "expected one folder, got " + str(runs)
    return relpath(runs[0], tensorboard_dir)


def test_revision(revision):
    model = BaBiRecurrentRelationalNet(True)
    model.load("%s/%s/best" % (model_dir, revision))
    batches = np.array(model.test_batches())

    logits = np.concatenate(batches[:, 0], axis=1)  # (5, 40k, 177)
    answers = np.concatenate(batches[:, 1], axis=0)  # (40k, )
    task_indices = np.concatenate(batches[:, 2], axis=0)  # (40k, )

    step, wt, acc_1M = get_1M_acc(revision)
    print(revision)
    print(step)
    print(acc_1M)
    for i in range(20):
        idx = task_indices == i
        expected = answers[idx]
        actual = np.argmax(logits[-1, idx, :], axis=1)
        acc = np.mean(expected == actual)
        print(acc)


def get_1M_acc(revision):
    run_name = get_run_name(revision)
    scalars = extract_scalars(run_name, 'steps/' + str(BaBiRecurrentRelationalNet.n_steps - 1) + '/tasks/avg')
    ew = 0.95
    ewma_acc = scalars[0][2]
    step = 0
    wt = 0
    for wt, step, acc in scalars:
        if step > int(1e6):
            return step, wt, ewma_acc
        ewma_acc = ew * ewma_acc + (1 - ew) * acc

    return step, wt, ewma_acc


def main():
    print("Testing revisions...")
    for r in test_revisions:
        test_revision(r)


if __name__ == '__main__':
    main()
