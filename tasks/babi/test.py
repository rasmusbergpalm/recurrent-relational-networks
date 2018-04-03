import glob
import os

import numpy as np
import requests
from os.path import relpath

from tasks.babi.rrn import BaBiRecurrentRelationalNet

test_revisions = ['ec566b2', 'c8c0176', '2a52711', 'ec016fc', '4898860', '208b4a9', '63e0ff8', '0b0f4d2']
model_dir = '/home/rapal/runs'
tensorboard_dir = os.environ.get('TENSORBOARD_DIR') + '/bAbI/debug/'
tensorboard_url = "http://localhost:6007/data/plugin/scalars/scalars"


def extract_scalars(run, tag):
    # ?run=ec566b2%2Ftest%2Fec566b2%20babi%20ablation%2C%20no%20qf%20encoding&tag=steps%2F2%2Ftasks%2Favg&format=json
    print(run, tag)
    response = requests.get(tensorboard_url, params={"run": run, "tag": tag, "format": "json"})
    print(response.status_code)
    print(response.content)
    return response.json()


def get_run_name(revision):
    runs = glob.glob(tensorboard_dir + revision + '/test/' + revision + '*')
    assert len(runs) == 1, "expected one folder, got " + str(runs)
    return relpath(runs[0], tensorboard_dir)


def test_revision(revision):
    print(revision)
    step, wt, acc_1M = get_1M_acc(revision)
    print(step, wt, acc_1M)

    model = BaBiRecurrentRelationalNet(True)
    model.load("%s/%s/best" % (model_dir, revision))
    batches = np.array(model.test_batches())

    logits = np.concatenate(batches[:, 0], axis=1)  # (5, 40k, 177)
    answers = np.concatenate(batches[:, 1], axis=0)  # (40k, )
    task_indices = np.concatenate(batches[:, 2], axis=0)  # (40k, )

    print(revision)
    step, wt, acc_1M = get_1M_acc(revision)
    print(step, wt, acc_1M)

    for i in range(20):
        idx = task_indices == i
        expected = answers[idx]
        actual = np.argmax(logits[-1, idx, :], axis=1)
        acc = np.mean(expected == actual)
        print(acc)


def get_1M_acc(revision):
    run_name = get_run_name(revision)
    scalars = extract_scalars(run_name, 'steps/' + str(BaBiRecurrentRelationalNet.n_steps - 1) + '/tasks/avg')
    momentum = 0.95
    ewma_acc = scalars[0][2]
    step = 0
    wt = 0
    for wt, step, acc in scalars:
        if step > int(1e6):
            return ewma_acc
        ewma_acc = momentum * ewma_acc + (1 - momentum) * acc
    return step, wt, ewma_acc


def main():
    print("Testing revisions...")
    for r in test_revisions:
        test_revision(r)


if __name__ == '__main__':
    main()
