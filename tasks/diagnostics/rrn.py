import os

import matplotlib
import numpy as np
import tensorflow as tf
from tensorboard.plugins.image.summary import pb as ipb
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.data import Dataset

import util
from message_passing import message_passing
from model import Model
from tasks.diagnostics.data import switch_targets_paths, TSP, Greedy

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DiagnosticRRN(Model):
    batch_size = 32
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    n = 16
    data = Greedy(n)
    n_steps = 16
    n_hidden = 32

    def __init__(self):
        super().__init__()
        self.name = "%s %s" % (self.revision, self.message)

        print("Building graph...")
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(2e-4)

        iterator = self._iterator(self.data)
        edges = [(i, j) for i in range(self.n) for j in range(self.n)]
        edges = tf.constant([(i + (b * self.n), j + (b * self.n)) for b in range(self.batch_size) for i, j in edges], tf.int32)
        edge_features = tf.zeros((tf.shape(edges)[0], 1))

        self.cities, self.indices, self.targets, self.paths = iterator.get_next()

        n_nodes = self.n * self.batch_size
        cities = tf.reshape(self.cities, (n_nodes, 2))
        indices = tf.one_hot(tf.reshape(self.indices, (n_nodes,)), self.n)
        targets = tf.reshape(self.targets, (n_nodes,))

        def mlp(x, scope):
            with tf.variable_scope(scope):
                for i in range(3):
                    x = layers.fully_connected(x, self.n_hidden)
                return layers.fully_connected(x, self.n_hidden, activation_fn=None)

        x = tf.concat([cities, indices], axis=1)
        x = mlp(x, 'pre')

        with tf.variable_scope('steps'):
            self.outputs = []
            losses = []
            x0 = x
            lstm_cell = LSTMCell(self.n_hidden)
            state = lstm_cell.zero_state(n_nodes, tf.float32)
            for step in range(self.n_steps):
                x = message_passing(x, edges, edge_features, lambda x: mlp(x, 'message-fn'))
                x = mlp(tf.concat([x, x0], axis=1), 'post')
                x, state = lstm_cell(x, state)

                logits = layers.fully_connected(x, num_outputs=(self.n), activation_fn=None, scope='logits')
                out = tf.argmax(tf.reshape(logits, (self.batch_size, (self.n), (self.n))), axis=2)
                self.outputs.append(out)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits) / tf.log(2.))
                losses.append(loss)

                tf.summary.scalar('steps/%d/loss' % step, loss)
                tf.get_variable_scope().reuse_variables()

        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', self.loss)

        optimizer = tf.train.AdamOptimizer()
        self.train_step = optimizer.minimize(self.loss, self.global_step)

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        util.print_vars(tf.trainable_variables())

        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/greedy/%s/train/%s' % (self.revision, self.name), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/greedy/%s/test/%s' % (self.revision, self.name), self.session.graph)
        self.summaries = tf.summary.merge_all()

    def train_batch(self):
        _, loss = self.session.run([self.train_step, self.loss])
        return loss

    def val_batch(self):
        loss, summaries, step, outputs, cities, paths = self.session.run([self.loss, self.summaries, self.global_step, self.outputs, self.cities, self.paths])
        self._write_summaries(self.test_writer, summaries, cities, outputs, paths, step)
        return loss

    def save(self, name):
        self.saver.save(self.session, name)

    def load(self, name):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)

    def _iterator(self, data):
        return Dataset.from_generator(
            data.sample_generator,
            data.output_types(),
            data.output_shapes()
        ).batch(self.batch_size).prefetch(1).make_one_shot_iterator()

    def _plot_paths(self, cities, expected, actual):
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(cities[:, 0], cities[:, 1])
        for i, txt in enumerate(range(len(cities))):
            plt.annotate(txt, (cities[i, 0], cities[i, 1]))
        plt.plot(cities[expected, 0], cities[expected, 1], color=(0, 0, 1., 0.5), markersize=0, label='expected')
        plt.plot(cities[actual, 0], cities[actual, 1], color=(1., 0, 0, 0.5), markersize=0, label='actual')
        plt.legend(loc='upper right')

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.axis('off')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

    def _write_summaries(self, writer, summaries, cities, outputs, paths, step):
        for t, b in enumerate(outputs):
            actual = switch_targets_paths(b[0])
            imgs = self._plot_paths(cities[0], paths[0], actual)
            paths_summary = ipb("paths/%d" % t, imgs[None])
            writer.add_summary(paths_summary, step)

        writer.add_summary(summaries, step)
        writer.flush()


if __name__ == '__main__':
    m = DiagnosticRRN(TSP)
    print(m.train_batch())
    print(m.val_batch())
