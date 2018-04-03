import os
import glob
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer

from tasks.babi.rrn import BaBiRecurrentRelationalNet

test_revisions = ['ec566b2', 'c8c0176', '2a52711', 'ec016fc', '4898860', '208b4a9', '63e0ff8', '0b0f4d2']
model_dir = '/home/rapal/runs'
tensorboard_dir = os.environ.get('TENSORBOARD_DIR') + '/bAbI/debug/'


def extract_scalars(multiplexer, run, tag):
    """Extract tabular data from the scalars at a given run and tag.
    The result is a list of 3-tuples (wall_time, step, value).
    """
    tensor_events = multiplexer.Tensors(run, tag)
    return [
        (event.wall_time, event.step, tf.make_ndarray(event.tensor_proto).item())
        for event in tensor_events
    ]


def create_multiplexer(logdir):
    multiplexer = event_multiplexer.EventMultiplexer(tensor_size_guidance={'scalars': 1000})
    multiplexer.AddRunsFromDirectory(logdir)
    multiplexer.Reload()
    return multiplexer


def get_run_name(revision):
    runs = glob.glob(tensorboard_dir + revision + '/test/' + revision + '*')
    assert len(runs) == 1, "expected one folder, got " + str(runs)
    return runs[0]


def test_revision(revision, multiplexer):
    model = BaBiRecurrentRelationalNet(True)
    model.load("%s/%s/best" % (model_dir, revision))
    batches = np.array(model.test_batches())

    logits = np.concatenate(batches[:, 0], axis=1)  # (5, 40k, 177)
    answers = np.concatenate(batches[:, 1], axis=0)  # (40k, )
    task_indices = np.concatenate(batches[:, 2], axis=0)  # (40k, )

    print(revision)
    step, wt, acc_1M = get_1M_acc(multiplexer, revision)
    print(step, wt, acc_1M)

    for i in range(20):
        idx = task_indices == i
        expected = answers[idx]
        actual = np.argmax(logits[-1, idx, :], axis=1)
        acc = np.mean(expected == actual)
        print(acc)


def get_1M_acc(multiplexer, revision):
    run_name = get_run_name(revision)
    scalars = extract_scalars(multiplexer, run_name, 'steps/' + str(BaBiRecurrentRelationalNet.n_steps) + '/tasks/avg')
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
    print("Creating multiplexer...")
    multiplexer = create_multiplexer(tensorboard_dir)
    print("Testing revisions...")
    for r in test_revisions:
        test_revision(r, multiplexer)


if __name__ == '__main__':
    main()
