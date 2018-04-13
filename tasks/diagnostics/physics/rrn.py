import os

import matplotlib
import networkx as nx
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.data import Dataset

import util
from message_passing import message_passing
from model import Model
from tensorboard.plugins.image.summary import pb as ipb
from tasks.diagnostics.physics.data import NBody

matplotlib.use('Agg')


class PhysicsRRN(Model):
    number = 1
    batch_size = 64
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    n_steps = 8
    n_hidden = 16
    n = 3
    devices = util.get_devices()

    def __init__(self):
        super().__init__()
        self.name = "%s %s" % (self.revision, self.message)

        print("Building graph...")
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.is_training_ph = tf.placeholder(bool, name='is_training')
        self.data = NBody()
        bs = self.batch_size
        regularizer = layers.l2_regularizer(0.)

        iterator = self._iterator(lambda: self.data.sample_generator(), self.output_types(), self.output_shapes())

        def mlp(x, scope, n_hid=self.n_hidden, n_out=self.n_hidden, keep_prob=1.0):
            with tf.variable_scope(scope):
                for i in range(3):
                    x = layers.fully_connected(x, n_hid, weights_regularizer=regularizer)
                x = layers.dropout(x, keep_prob=keep_prob, is_training=self.is_training_ph)
                return layers.fully_connected(x, n_out, weights_regularizer=regularizer, activation_fn=None)

        self.inputs = iterator.get_next()  # (bs, 3, 128, 4)
        self.expected = tf.transpose(self.inputs[:, :, :, :2], (2, 0, 1, 3))

        self.inputs = tf.reshape(self.inputs, (bs * self.n, 128, 4))  # (bs*3, 128, 4)

        edges = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]
        edges = [(i + (b * self.n), j + (b * self.n)) for b in range(bs) for i, j in edges]
        assert len(list(nx.connected_component_subgraphs(nx.Graph(edges)))) == bs
        edges = tf.constant(edges, tf.int32)  # (bs*3*3, 2)

        x = self.inputs[:, 0, :]  # (bs*3, 4)
        self.targets = self.inputs[:, :, :2]  # (bs*3, 128, 2) positions

        edge_features = tf.zeros((tf.shape(edges)[0], 1))

        outputs = []
        losses = []
        with tf.variable_scope('steps'):
            for step in range(self.n_steps):
                x = message_passing(x, edges, edge_features, lambda x: mlp(x, 'message-fn', n_out=4))
                out = mlp(x, "out", n_out=2)  # (bs*3, 2)

                outputs.append(out)  # (n_steps, bs*3, 2)
                loss = tf.losses.mean_squared_error(labels=self.targets[:, step], predictions=out)
                losses.append(loss)

                tf.get_variable_scope().reuse_variables()

        losses = tf.reduce_mean(losses)
        outputs = tf.concat(outputs, axis=0)  # (n_steps, bs*3, 2)
        self.outputs = tf.reshape(outputs, (self.n_steps, bs, self.n, 2))

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('reg_loss', reg_loss)

        self.loss = losses + reg_loss
        tf.summary.scalar('loss', self.loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)
            self.train_step = self.optimizer.apply_gradients(gvs, global_step=self.global_step)

        """
        for i, (g, v) in enumerate(gvs):
            tf.summary.histogram("grads/%03d/%s" % (i, v.name), g)
            tf.summary.histogram("vars/%03d/%s" % (i, v.name), v)
            tf.summary.histogram("g_ratio/%03d/%s" % (i, v.name), tf.log(tf.abs(g) + 1e-8) - tf.log(tf.abs(v) + 1e-8))
        """

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        util.print_vars(tf.trainable_variables())

        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/physics/%s/train/%s' % (self.revision, self.name), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/physics/%s/test/%s' % (self.revision, self.name), self.session.graph)
        self.summaries = tf.summary.merge_all()

    def train_batch(self):
        step = self.session.run(self.global_step)
        if step % 1000 == 0:
            _, loss, summaries = self.session.run([self.train_step, self.loss, self.summaries], {self.is_training_ph: True})
            self.train_writer.add_summary(summaries, step)
        else:
            _, loss = self.session.run([self.train_step, self.loss], {self.is_training_ph: True})
        return loss

    def val_batch(self):
        loss, summaries, step, outputs, expected = self.session.run([self.loss, self.summaries, self.global_step, self.outputs, self.expected], {self.is_training_ph: False})
        self.write_summaries(self.test_writer, summaries, expected, outputs)
        return loss

    def save(self, name):
        self.saver.save(self.session, name)

    def load(self, name):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)

    def _iterator(self, generator, output_types, output_shapes):
        return Dataset.from_generator(
            generator,
            output_types,
            output_shapes
        ).batch(self.batch_size).prefetch(1).make_one_shot_iterator()

    def output_types(self):
        return tf.float32

    def output_shapes(self):
        return (3, 128, 4)

    def write_summaries(self, writer, summaries, targets, outputs):
        writer.add_summary(summaries)
        img = self.data.trace_diff(targets[:, 0, :, :], outputs[:, 0, :, :])
        img_summary = ipb("trace", img[None])
        writer.add_summary(img_summary)
        writer.flush()

if __name__ == '__main__':
    m = PhysicsRRN()
    print(m.train_batch())
    print(m.val_batch())
