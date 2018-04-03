import numpy as np

from tasks.babi.rrn import BaBiRecurrentRelationalNet

test_revisions = ['ec566b2', 'c8c0176', '2a52711', 'ec016fc', '4898860', '208b4a9', '63e0ff8', '0b0f4d2']

model_dir = '/home/rapal/runs'
n_steps = BaBiRecurrentRelationalNet.n_steps


def test_revision(revision):
    model = BaBiRecurrentRelationalNet(True)
    model.load("%s/%s/best" % (model_dir, revision))
    batches = np.array(model.test_batches())

    logits = np.concatenate(batches[:, 0], axis=1)  # (5, 40k, 177)
    answers = np.concatenate(batches[:, 1], axis=0)  # (40k, )
    task_indices = np.concatenate(batches[:, 2], axis=0)  # (40k, )

    print(revision)
    for i in range(20):
        idx = task_indices == i
        expected = answers[idx]
        actual = np.argmax(logits[-1, idx, :], axis=1)
        acc = np.mean(expected == actual)
        print(acc)


for r in test_revisions:
    test_revision(r)
