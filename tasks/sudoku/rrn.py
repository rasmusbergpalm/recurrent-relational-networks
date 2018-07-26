import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.rnn import LSTMCell

import util
from message_passing import message_passing
from model import Model
from tasks.sudoku.data import sudoku


class SudokuRecurrentRelationalNet(Model):
    devices = util.get_devices()
    batch_size = 64
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    emb_size = 16
    n_steps = 32
    edge_keep_prob = 1.0
    n_hidden = 96
    edges = 'sudoku'

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

            if self.edges == 'sudoku':
                edges = self.sudoku_edges()
            elif self.edges == 'full':
                edges = [(i, j) for i in range(81) for j in range(81) if not i == j]
            else:
                raise ValueError('edges must be sudoku or full')

            edge_indices = tf.constant([(i + (b * 81), j + (b * 81)) for b in range(self.batch_size) for i, j in edges], tf.int32)
            n_edges = tf.shape(edge_indices)[0]
            edge_features = tf.zeros((n_edges, 1), tf.float32)
            positions = tf.constant([[(i, j) for i in range(9) for j in range(9)] for b in range(self.batch_size)], tf.int32)  # (bs, 81, 2)
            rows = 0. * layers.embed_sequence(positions[:, :, 0], 9, self.emb_size, scope='row-embeddings', unique=True)  # bs, 81, emb_size
            cols = 0. * layers.embed_sequence(positions[:, :, 1], 9, self.emb_size, scope='cols-embeddings', unique=True)  # bs, 81, emb_size

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

                        def mlp(x, scope):
                            with tf.variable_scope(scope):
                                for i in range(3):
                                    x = layers.fully_connected(x, self.n_hidden, weights_regularizer=regularizer)
                                return layers.fully_connected(x, self.n_hidden, weights_regularizer=regularizer, activation_fn=None)

                        x = mlp(x, 'pre-fn')
                        x0 = x
                        n_nodes = tf.shape(x)[0]
                        outputs = []
                        log_losses = []
                        with tf.variable_scope('steps'):
                            lstm_cell = LSTMCell(self.n_hidden)
                            state = lstm_cell.zero_state(n_nodes, tf.float32)

                            for step in range(self.n_steps):
                                x = message_passing(x, edge_indices, edge_features, lambda x: mlp(x, 'message-fn'), edge_keep_prob)
                                x = mlp(tf.concat([x, x0], axis=1), 'post-fn')
                                x, state = lstm_cell(x, state)

                                with tf.variable_scope('graph-sum'):
                                    out = tf.reshape(layers.fully_connected(x, num_outputs=10, activation_fn=None), (-1, 81, 10))
                                    outputs.append(out)
                                    log_losses.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answers, logits=out)))

                                tf.get_variable_scope().reuse_variables()

                        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        loss = avg_n(log_losses) + reg_loss

                        towers.append({
                            'loss': loss,
                            'grads': self.optimizer.compute_gradients(loss),
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

            self.train_writer = tf.summary.FileWriter('/tmp/tensorboard/sudoku/%s/train/%s' % (self.revision, self.name), self.session.graph)
            self.test_writer = tf.summary.FileWriter('/tmp/tensorboard/sudoku/%s/test/%s' % (self.revision, self.name), self.session.graph)
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
