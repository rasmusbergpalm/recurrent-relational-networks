import glob
import os
from os.path import relpath

import numpy as np
import requests

from tasks.babi.rrn import BaBiRecurrentRelationalNet

experiments = {
    'baseline': ['8a16a10', 'c94d161', '4dafa06', '666db79', '140f16a', 'c1c99a2', '17446a7', '0f31737', '5fe495d', '3c4576f', '9ffa0b8', '81eee06', '8089d12', 'a460279', 'b5bd840'],
    'linear qf': ['4e1d56c', 'ac440ad', '889fcb4', '7c0c35e', 'e144ca9', 'a58ea1a', '5273506', 'dd91e9a'],
    'only f': ['ec566b2', 'c8c0176', '2a52711', 'ec016fc', '4898860', '208b4a9', '63e0ff8', '0b0f4d2'],
    'no dropout': ['c458dbe', 'ed0900f', 'db7e50d', 'c26a2a5', '48d6025', '7a8e4ba', '1623750', '93f6d7f']
}
test = 'no dropout'
n_steps = range(3)

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

    steps = []
    for n_step in n_steps:
        step, wt, acc_1M = get_1M_acc(revision, n_step)

        result = {
            'revision': revision,
            'step': step,
            'acc_1M': acc_1M,
            'tasks': []
        }

        for i in range(20):
            idx = task_indices == i
            expected = answers[idx]
            actual = np.argmax(logits[n_step, idx, :], axis=1)
            acc = np.mean(expected == actual)
            result['tasks'].append(acc)

        steps.append(result)

    return steps


def get_1M_acc(revision, n_step):
    run_name = get_run_name(revision)

    scalars = extract_scalars(run_name, 'steps/' + str(n_step) + '/tasks/avg')
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
    print("Testing...")
    revisions = [test_revision(r) for r in experiments[test]]

    for i in n_steps:
        print("------------------ STEP %d ------------------" % i)
        for revision in revisions:
            r = revision[i]
            print("%s,%d,%f,%s" % (r['revision'], r['step'], r['acc_1M'], ",".join(map(str, r['tasks']))))


if __name__ == '__main__':
    main()
