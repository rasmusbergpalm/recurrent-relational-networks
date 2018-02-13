import os
import tensorflow as tf
from tensorflow.python.data import Dataset

from message_passing import message_passing
from model import Model
from tasks.diagnostics.greedy.data import Greedy
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import LSTMCell
import util


class GreedyRRN(Model):
    n = 9
    batch_size = 32
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    n_steps = n
    n_hidden = 32

    def __init__(self):
        super().__init__()
        self.name = "%s %s" % (self.revision, self.message)

        print("Building graph...")
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer()

        iterator = self._iterator(Greedy(self.n))
        edges = [(i, j) for i in range(self.n) for j in range(self.n)]
        edges = tf.constant([(i + (b * self.n), j + (b * self.n)) for b in range(self.batch_size) for i, j in edges], tf.int32)
        edge_features = tf.zeros((tf.shape(edges)[0], 1))

        cities, indices, targets, paths = iterator.get_next()

        n_nodes = self.n * self.batch_size
        cities = tf.reshape(cities, (n_nodes, 2))
        indices = tf.one_hot(tf.reshape(indices, (n_nodes,)), self.n)
        targets = tf.reshape(targets, (n_nodes,))

        def mlp(x, scope):
            with tf.variable_scope(scope):
                for i in range(3):
                    x = layers.fully_connected(x, self.n_hidden)
                return layers.fully_connected(x, self.n_hidden, activation_fn=None)

        x = tf.concat([cities, indices], axis=1)
        x = mlp(x, 'pre')

        with tf.variable_scope('steps'):
            outputs = []
            losses = []
            x0 = x
            lstm_cell = LSTMCell(self.n_hidden)
            state = lstm_cell.zero_state(n_nodes, tf.float32)
            for step in range(self.n_steps):
                x = message_passing(x, edges, edge_features, lambda x: mlp(x, 'message-fn'))
                x = mlp(tf.concat([x, x0], axis=1), 'post')
                x, state = lstm_cell(x, state)

                out = layers.fully_connected(x, num_outputs=self.n, activation_fn=None, scope='out')
                outputs.append(out)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=out) / tf.log(2.))
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
        _, loss, summaries, step = self.session.run([self.train_step, self.loss, self.summaries, self.global_step])
        self.train_writer.add_summary(summaries, step)
        self.train_writer.flush()
        return loss

    def val_batch(self):
        loss, summaries, step = self.session.run([self.loss, self.summaries, self.global_step])
        self.test_writer.add_summary(summaries, step)
        self.test_writer.flush()

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


if __name__ == '__main__':
    m = GreedyRRN()
    print(m.train_batch())
