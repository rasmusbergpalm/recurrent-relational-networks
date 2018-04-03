import numpy as np

from tasks.babi.rrn import BaBiRecurrentRelationalNet

test_revisions = ['4e1d56c', 'ac440ad', '889fcb4', '7c0c35e', 'e144ca9', 'a58ea1a', '5273506', 'dd91e9a']

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
