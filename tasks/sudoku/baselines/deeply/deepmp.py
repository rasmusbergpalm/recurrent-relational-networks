import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.data import Iterator

import util
from model import Model
from tasks.sudoku.data import sudoku


def message_passing(nodes, edges, edge_features, message_fn, edge_keep_prob=1.0):
    """
    Pass messages between nodes and sum the incoming messages at each node.
    Implements equation 1 and 2 in the paper, i.e. m_{.j}^t &= \sum_{i \in N(j)} f(h_i^{t-1}, h_j^{t-1})

    :param nodes: (n_nodes, n_features) tensor of node hidden states.
    :param edges: (n_edges, 2) tensor of indices (i, j) indicating an edge from nodes[i] to nodes[j].
    :param edge_features: features for each edge. Set to zero if the edges don't have features.
    :param message_fn: message function, will be called with input of shape (n_edges, 2*n_features + edge_features). The output shape is (n_edges, n_outputs), where you decide the size of n_outputs
    :param edge_keep_prob: The probability by which edges are kept. Basically dropout for edges. Not used in the paper.
    :return: (n_nodes, n_output) Sum of messages arriving at each node.
    """
    n_nodes = tf.shape(nodes)[0]
    n_features = nodes.get_shape()[1].value
    n_edges = tf.shape(edges)[0]

    message_inputs = tf.gather(nodes, edges)  # n_edges, 2, n_features
    reshaped = tf.concat([tf.reshape(message_inputs, (-1, 2 * n_features)), edge_features], 1)
    messages = message_fn(reshaped)  # n_edges, n_output
    messages = tf.nn.dropout(messages, edge_keep_prob, noise_shape=(n_edges, 1))

    n_output = messages.get_shape()[1].value

    idx_i, idx_j = tf.split(edges, 2, 1)
    out_shape = (n_nodes, n_output)
    updates = tf.scatter_nd(idx_j, messages, out_shape)

    return updates, messages


class SudokuDeeplyLearnedMessages(Model):
    devices = util.get_devices()
    batch_size = 256 // len(devices)
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    emb_size = 16
    n_steps = 32
    edge_keep_prob = 1.0
    n_hidden = 96
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'

    def __init__(self, is_testing):
        super().__init__()
        self.is_testing = is_testing
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            regularizer = layers.l2_regularizer(1e-4)
            self.name = "%s %s" % (self.revision, self.message)
            self.train, self.valid, self.test = self.encode_data(sudoku())

            print("Building graph...")
            self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.global_step = tf.Variable(initial_value=0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)
            self.mode = tf.placeholder(tf.string)

            edges = self.sudoku_edges()
            edges = [(i + (b * 81), j + (b * 81)) for b in range(self.batch_size) for i, j in edges]
            ridx = [edges.index((j, i)) for i, j in edges]
            edge_indices = tf.constant(edges, tf.int32)
            n_edges = tf.shape(edge_indices)[0]

            positions = tf.constant([[(i, j) for i in range(9) for j in range(9)] for b in range(self.batch_size)], tf.int32)  # (bs, 81, 2)
            rows = layers.embed_sequence(positions[:, :, 0], 9, self.emb_size, scope='row-embeddings', unique=True)  # bs, 81, emb_size
            cols = layers.embed_sequence(positions[:, :, 1], 9, self.emb_size, scope='cols-embeddings', unique=True)  # bs, 81, emb_size

            def avg_n(x):
                return tf.reduce_mean(tf.stack(x, axis=0), axis=0)

            towers = []
            with tf.variable_scope(tf.get_variable_scope()):
                for device_nr, device in enumerate(self.devices):
                    with tf.device('/cpu:0'):

                        if self.is_testing:
                            (quizzes, answers), edge_keep_prob = self.test.get_next(), 1.0
                        else:
                            (quizzes, answers), edge_keep_prob = tf.cond(
                                tf.equal(self.mode, "train"),
                                true_fn=lambda: (self.train.get_next(), self.edge_keep_prob),
                                false_fn=lambda: (self.valid.get_next(), 1.0)
                            )

                        x = layers.embed_sequence(quizzes, 10, self.emb_size, scope='nr-embeddings', unique=True)  # bs, 81, emb_size
                        x = tf.concat([x, rows, cols], axis=2)
                        x = tf.reshape(x, (-1, 3 * self.emb_size))

                    with tf.device(device), tf.name_scope("device-%s" % device_nr):

                        def mlp(x, scope, n_out):
                            with tf.variable_scope(scope):
                                for i in range(3):
                                    x = layers.fully_connected(x, n_out, weights_regularizer=regularizer)
                                return layers.fully_connected(x, n_out, weights_regularizer=regularizer, activation_fn=None)

                        x = mlp(x, 'C1', self.n_hidden)
                        dependents = tf.zeros((n_edges, 10))
                        outputs = []
                        log_losses = []
                        with tf.variable_scope('steps'):
                            for step in range(self.n_steps):
                                # M_F = c2(c1(x, p), c1(x, N_F\p), d_pF)
                                # d_pF = sum_{q \in N_F\p} (M_F)
                                # p(y_p|x) = softmax(sum(M_F))

                                logits, messages = message_passing(x, edge_indices, dependents, lambda x: mlp(x, 'C2', 10))
                                dependents = tf.gather(logits, edge_indices[:, 0]) - tf.gather(messages, ridx)
                                out = tf.reshape(logits, (-1, 81, 10))
                                outputs.append(out)
                                log_losses.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answers, logits=out)))
                                tf.get_variable_scope().reuse_variables()

                        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        loss = log_losses[-1] + reg_loss

                        towers.append({
                            'loss': loss,
                            'grads': [(tf.clip_by_value(g, -10.0, 10.0), v) for g, v in self.optimizer.compute_gradients(loss)],
                            'log_losses': tf.stack(log_losses),  # (n_steps, 1)
                            'quizzes': quizzes,  # (bs, 81, 10)
                            'answers': answers,  # (bs, 81, 10)
                            'outputs': tf.stack(outputs)  # n_steps, bs, 81, 10
                        })

                        tf.get_variable_scope().reuse_variables()

            self.loss = avg_n([t['loss'] for t in towers])
            self.out = tf.concat([t['outputs'] for t in towers], axis=1)  # n_steps, bs, 81, 10
            self.predicted = tf.cast(tf.argmax(self.out, axis=3), tf.int32)
            self.answers = tf.concat([t['answers'] for t in towers], axis=0)
            self.quizzes = tf.concat([t['quizzes'] for t in towers], axis=0)

            tf.summary.scalar('losses/total', self.loss)
            tf.summary.scalar('losses/reg', reg_loss)
            log_losses = avg_n([t['log_losses'] for t in towers])

            for step in range(self.n_steps):
                equal = tf.equal(self.answers, self.predicted[step])

                digit_acc = tf.reduce_mean(tf.to_float(equal))
                tf.summary.scalar('steps/%d/digit-acc' % step, digit_acc)

                puzzle_acc = tf.reduce_mean(tf.to_float(tf.reduce_all(equal, axis=1)))
                tf.summary.scalar('steps/%d/puzzle-acc' % step, puzzle_acc)

                tf.summary.scalar('steps/%d/losses/log' % step, log_losses[step])

            avg_gradients = util.average_gradients([t['grads'] for t in towers])
            self.train_step = self.optimizer.apply_gradients(avg_gradients, global_step=self.global_step)

            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            util.print_vars(tf.trainable_variables())

            self.train_writer = tf.summary.FileWriter(self.tensorboard_dir + '/sudoku/%s/train/%s' % (self.revision, self.name), self.session.graph)
            self.test_writer = tf.summary.FileWriter(self.tensorboard_dir + '/sudoku/%s/test/%s' % (self.revision, self.name), self.session.graph)
            self.summaries = tf.summary.merge_all()

    def sudoku_edges(self):
        def cross(a):
            return [(i, j) for i in a.flatten() for j in a.flatten() if not i == j]

        idx = np.arange(81).reshape(9, 9)
        rows, columns, squares = [], [], []
        for i in range(9):
            rows += cross(idx[i, :])
            columns += cross(idx[:, i])
        for i in range(3):
            for j in range(3):
                squares += cross(idx[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])
        return list(set(rows + columns + squares))

    def encode_data(self, data) -> (Iterator, Iterator, Iterator):
        def encode(samples, n_repeat):
            def parse(x):
                return list(map(int, list(x)))

            encoded = [(parse(q), parse(a)) for q, a in samples]
            q, a = zip(*encoded)
            q, a = np.array(q, np.int32), np.array(a, np.int32)
            return Dataset.from_tensor_slices((q, a)).shuffle(self.batch_size * 10).repeat(n_repeat).batch(self.batch_size).make_one_shot_iterator()

        print("Encoding data...")
        train = encode(data.train, -1)
        valid = encode(data.valid, -1)
        test = encode(data.test, 1)

        return train, valid, test

    def save(self, name):
        self.saver.save(self.session, name)

    def load(self, name):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)

    def train_batch(self):
        _, _loss, _logits, _summaries, _step = self.session.run([self.train_step, self.loss, self.out, self.summaries, self.global_step], {self.mode: 'train'})
        if _step % 1000 == 0:
            self.train_writer.add_summary(_summaries, _step)

        return _loss

    def test_batch(self):
        _quizzes, _logits, _answers = self.session.run([self.quizzes, self.out, self.answers], {self.mode: 'foo'})
        return _quizzes, _logits, _answers

    def val_batch(self):
        _loss, _predicted, _answers, _summaries, _step = self.session.run([self.loss, self.predicted, self.answers, self.summaries, self.global_step], {self.mode: 'valid'})
        self.test_writer.add_summary(_summaries, _step)
        return _loss
