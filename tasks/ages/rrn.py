import os

import matplotlib
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorboard.plugins.scalar.summary import pb as spb
from tensorflow.contrib import layers
from tensorflow.python.data import Dataset

import util
from message_passing import message_passing
from model import Model
from tasks.ages.data import Ages

matplotlib.use('Agg')


class AgesRRN(Model):
    number = 1
    batch_size = 128
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    n_steps = 8
    n_hidden = 128
    devices = util.get_devices()

    def __init__(self):
        super().__init__()
        self.name = "%s %s" % (self.revision, self.message)

        print("Building graph...")
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.mode = tf.placeholder(tf.string, name='mode')
        self.is_training = tf.equal(self.mode, "train")
        self.data = Ages()

        regularizer = layers.l2_regularizer(0.)

        iterator = self._iterator(self.data)

        n_nodes = 8

        def mlp(x, scope, n_hid=self.n_hidden, n_out=self.n_hidden, keep_prob=1.0):
            with tf.variable_scope(scope):
                for i in range(1):
                    x = layers.fully_connected(x, n_hid, weights_regularizer=regularizer)
                x = layers.dropout(x, keep_prob=keep_prob, is_training=self.is_training)
                return layers.fully_connected(x, n_out, weights_regularizer=regularizer, activation_fn=None)

        def forward(sources, targets, types, diffs, question, answers):
            """

            :param sources: (bs, n)
            :param targets: (bs, n)
            :param types: (bs, n)
            :param diffs: (bs, n)
            :param question: (bs, 1)
            :param answers: (bs, 1)
            :param n_jumps: (bs, 1)
            :return:
            """
            bs = self.batch_size
            segment_ids = sum([[i] * n_nodes for i in range(bs)], [])

            edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
            edges = [(i + (b * n_nodes), j + (b * n_nodes)) for b in range(bs) for i, j in edges]
            assert len(list(nx.connected_component_subgraphs(nx.Graph(edges)))) == bs
            edges = tf.constant(edges, tf.int32)  # (bs*8*8, 2)

            sources = tf.reshape(tf.one_hot(sources, n_nodes), (bs * n_nodes, n_nodes))
            targets = tf.reshape(tf.one_hot(targets, n_nodes), (bs * n_nodes, n_nodes))
            types = tf.reshape(tf.one_hot(types, 3), (bs * n_nodes, 3))
            diffs = tf.reshape(tf.one_hot(diffs, 100), (bs * n_nodes, 100))

            question = tf.one_hot(question, n_nodes)
            question = tf.gather(question, segment_ids)  # (bs*n_nodes, n_nodes)

            x = tf.concat([sources, targets, types, diffs, question], axis=1)
            x = mlp(x, 'pre-fn')

            edge_features = tf.zeros_like(edges, tf.float32)

            with tf.variable_scope('steps'):
                outputs = []
                losses = []
                h = x

                for step in range(self.n_steps):
                    h_p = h
                    m = message_passing(h, edges, edge_features, lambda x: mlp(x, 'message-fn'))
                    h = mlp(tf.concat([x, h_p, m], axis=1), 'node-fn')

                    logits = tf.unsorted_segment_sum(h, segment_ids, bs)
                    logits = mlp(logits, "out", n_out=100, keep_prob=0.5)

                    out = tf.argmax(logits, axis=1)
                    outputs.append(out)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answers, logits=logits) / tf.log(2.))
                    losses.append(loss)

                    tf.get_variable_scope().reuse_variables()

            return losses, outputs

        sources, targets, types, diffs, question, self.answers, self.n_jumps = iterator.get_next()

        log_losses, outputs = forward(sources, targets, types, diffs, question, self.answers)

        log_losses = tf.reduce_mean(log_losses)
        self.outputs = outputs  # (steps, bs)

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('reg_loss', reg_loss)

        self.loss = tf.reduce_mean(log_losses) + reg_loss
        tf.summary.scalar('loss', self.loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)
            self.train_step = self.optimizer.apply_gradients(gvs, global_step=self.global_step)

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        util.print_vars(tf.trainable_variables())

        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/ages/%s/train/%s' % (self.revision, self.name), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/ages/%s/test/%s' % (self.revision, self.name), self.session.graph)
        self.summaries = tf.summary.merge_all()

    def train_batch(self):
        step = self.session.run(self.global_step)
        if step % 1000 == 0:
            _, loss, summaries = self.session.run([self.train_step, self.loss, self.summaries], {self.mode: "train"})
            self.train_writer.add_summary(summaries, step)
        else:
            _, loss = self.session.run([self.train_step, self.loss], {self.mode: "train"})
        return loss

    def val_batch(self):
        loss, summaries, step, jumps, answers, outputs = self.session.run([self.loss, self.summaries, self.global_step, self.n_jumps, self.answers, self.outputs], {self.mode: "test"})
        self._write_summaries(self.test_writer, summaries, outputs, answers, jumps, step)
        return loss

    def save(self, name):
        self.saver.save(self.session, name)

    def load(self, name):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)

    def _iterator(self, data):
        types, shapes = data.types_and_shapes()
        return Dataset.from_generator(
            data.generator,
            types,
            shapes
        ).batch(self.batch_size).prefetch(1).make_one_shot_iterator()

    def _write_summaries(self, writer, summaries, outputs, answers, jumps, step):
        accs = self.compute_acc(jumps, outputs, answers)
        for s in range(self.n_steps):
            for j, v in accs[s].items():
                writer.add_summary(spb("acc/%d/%d" % (s, j), v), step)

        writer.add_summary(summaries, step)
        writer.flush()

    def compute_acc(self, jumps, outputs, targets):
        accs = []
        for t in range(self.n_steps):
            jumps_acc = {}
            equal = outputs[t] == targets
            for i in range(7):
                jumps_i = jumps == i
                if any(jumps_i):
                    acc = np.mean(equal[jumps_i])
                    jumps_acc[i] = acc
            accs.append(jumps_acc)
        return accs


if __name__ == '__main__':
    m = AgesRRN()
    print(m.val_batch())
    print(m.train_batch())
