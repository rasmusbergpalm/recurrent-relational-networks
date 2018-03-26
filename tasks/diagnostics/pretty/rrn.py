import os

import matplotlib
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorboard.plugins.image.summary import pb as ipb
from tensorboard.plugins.scalar.summary import pb as spb
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.data import Dataset

import util
from message_passing import message_passing
from model import Model
from tasks.diagnostics.pretty.data import PrettyClevr, fig2array

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PrettyRRN(Model):
    number = 4
    batch_size = 128
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    n_objects = 8
    data = PrettyClevr()
    n_steps = 4
    n_hidden = 128
    devices = util.get_devices()

    def __init__(self):
        super().__init__()
        self.name = "%s %s" % (self.revision, self.message)

        print("Building graph...")
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.is_training_ph = tf.placeholder(bool, name='is_training')

        regularizer = layers.l2_regularizer(1e-4)

        train_iterator = self._iterator(self.data.train_generator, self.data.output_types(), self.data.output_shapes())
        dev_iterator = self._iterator(self.data.dev_generator, self.data.output_types(), self.data.output_shapes())
        n_nodes = 8
        n_anchors_targets = len(self.data.i2s)

        def mlp(x, scope, n_hid=self.n_hidden, n_out=self.n_hidden, keep_prob=1.0):
            with tf.variable_scope(scope):
                for i in range(3):
                    x = layers.fully_connected(x, n_hid, weights_regularizer=regularizer)
                x = layers.dropout(x, keep_prob=keep_prob, is_training=self.is_training_ph)
                return layers.fully_connected(x, n_out, weights_regularizer=regularizer, activation_fn=None)

        def forward(img, anchors, n_jumps, targets, positions, colors, markers):
            """
            :param img: (bs, 128, 128, 3)
            :param anchors: (bs,)
            :param n_jumps: (bs,)
            :param targets: (bs,)
            :param positions: (bs, 8, 2)
            :param colors: (bs, 8)
            """
            bs = self.batch_size // len(self.devices)
            segment_ids = sum([[i] * n_nodes for i in range(bs)], [])

            edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
            edges = [(i + (b * n_nodes), j + (b * n_nodes)) for b in range(bs) for i, j in edges]
            assert len(list(nx.connected_component_subgraphs(nx.Graph(edges)))) == bs
            edges = tf.constant(edges, tf.int32)  # (bs*8*8, 2)
            """
            
            x = ((1. - tf.to_float(img) / 255.) - 0.5)  # (bs, h, w, 3)
            with tf.variable_scope('encoder'):
                for i in range(5):
                    x = layers.conv2d(x, num_outputs=self.n_hidden, kernel_size=3, stride=2)  # (bs, 4, 4, 128)
            x = tf.reshape(x, (bs * n_nodes, self.n_hidden))
            

            def dist(positions):
                expanded_a = tf.expand_dims(positions, 2)  # (bs, 8, 1, 2)
                expanded_b = tf.expand_dims(positions, 1)  # (bs, 1, 8, 2)
                return tf.sqrt(tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 3))  # (bs, 8, 8)
            """

            colors = tf.reshape(tf.one_hot(colors, 8), (bs * n_nodes, 8))
            markers = tf.reshape(tf.one_hot(markers - 8, 8), (bs * n_nodes, 8))
            positions = tf.reshape(positions, (bs * n_nodes, 2))

            distances = tf.gather(positions, edges)  # (n_edges, 2, 2)
            distances = tf.sqrt(tf.reduce_sum(tf.square(distances[:, 0] - distances[:, 1]), axis=1, keep_dims=True))  # (n_edges, 1)

            question = tf.one_hot(anchors, n_anchors_targets)
            question = tf.gather(question, segment_ids)

            x = tf.concat([positions, colors, markers, question], axis=1)
            x = mlp(x, 'pre')

            # logits = layers.fully_connected(x, n_anchors_targets, activation_fn=None, scope="logits")

            """
            n_edges = tf.shape(edges)[0]
            question = tf.concat([tf.one_hot(anchors, n_anchors_targets), tf.one_hot(n_jumps, self.n_objects)], axis=1)  # (bs, 24)
            question = tf.reshape(tf.tile(tf.expand_dims(question, 1), [1, n_nodes, 1]), [n_edges, 24])

            edge_features = tf.reshape(tf.concat([question, distances], axis=1), [n_edges, 25])
            """
            edge_features = distances

            with tf.variable_scope('steps'):
                outputs = []
                losses = []
                x0 = x
                lstm_cell = LSTMCell(self.n_hidden)

                """
                state = LSTMStateTuple(
                    tf.get_variable('LSTM/c_init', shape=(n_nodes * bs, self.n_hidden), dtype=tf.float32, initializer=tf.initializers.random_normal),
                    tf.get_variable('LSTM/h_init', shape=(n_nodes * bs, self.n_hidden), dtype=tf.float32, initializer=tf.initializers.random_normal)
                )
                """

                state = lstm_cell.zero_state(n_nodes * bs, tf.float32)

                for step in range(self.n_steps):
                    x = message_passing(x, edges, edge_features, lambda x: mlp(x, 'message-fn'))
                    x = mlp(tf.concat([x, x0], axis=1), 'post', keep_prob=0.5)
                    x = layers.batch_norm(x, is_training=self.is_training_ph, scope='BN')
                    x, state = lstm_cell(x, state)

                    logits = tf.unsorted_segment_sum(x, segment_ids, bs)
                    logits = mlp(logits, "out", n_out=n_anchors_targets, keep_prob=0.5)

                    out = tf.argmax(logits, axis=1)
                    outputs.append(out)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits) / tf.log(2.))
                    losses.append(loss)

                    tf.get_variable_scope().reuse_variables()

            return losses, outputs

        self.org_img, positions, colors, markers, self.anchors, self.n_jumps, self.targets = tf.cond(
            self.is_training_ph,
            true_fn=lambda: train_iterator.get_next(),
            false_fn=lambda: dev_iterator.get_next(),
        )

        log_losses, outputs = util.batch_parallel(forward, self.devices, img=self.org_img, anchors=self.anchors, n_jumps=self.n_jumps, targets=self.targets, positions=positions, colors=colors, markers=markers)
        log_losses = tf.reduce_mean(log_losses)
        self.outputs = tf.concat(outputs, axis=1)  # (splits, steps, bs)

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('reg_loss', reg_loss)

        self.loss = tf.reduce_mean(log_losses) + reg_loss
        tf.summary.scalar('loss', self.loss)

        gvs = self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)
        for g, v in gvs:
            tf.summary.histogram("grads/" + v.name, g)
            tf.summary.histogram("vars/" + v.name, v)
            tf.summary.histogram("g_ratio/" + v.name, g / (v + 1e-8))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.apply_gradients(gvs, global_step=self.global_step)

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        util.print_vars(tf.trainable_variables())

        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/pretty/%s/train/%s' % (self.revision, self.name), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/pretty/%s/test/%s' % (self.revision, self.name), self.session.graph)
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
        loss, summaries, step, img, anchors, jumps, targets, outputs = self.session.run([self.loss, self.summaries, self.global_step, self.org_img, self.anchors, self.n_jumps, self.targets, self.outputs], {self.is_training_ph: False})
        self._write_summaries(self.test_writer, summaries, img, anchors, jumps, targets, outputs, step)
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

    def _render(self, img, anchor, jump, target, outputs):
        remap = {'blue': 'b', 'green': 'g', 'red': 'r', 'cyan': 'c', 'magenta': 'm', 'yellow': 'y', 'black': 'k', 'gray': 'a'}
        fig = plt.figure(figsize=(2.56, 2.56), frameon=False)
        plt.imshow(img)
        outs = [self.data.i2s[output[0]] for output in outputs]
        out_str = "".join([remap[o] if o in remap else o for o in outs])
        title = "%s %d %s\n%s" % (self.data.i2s[anchor], jump, self.data.i2s[target], out_str)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        return fig2array(fig)

    def _write_summaries(self, writer, summaries, img, anchors, jumps, targets, outputs, step):
        for t in range(self.n_steps):
            equal = outputs[t] == targets
            for i in range(8):
                jumps_i = jumps == i
                if any(jumps_i):
                    acc = np.mean(equal[jumps_i])
                    writer.add_summary(spb("acc/%d/%d" % (t, i), acc), step)

        imgs = self._render(img[0], int(anchors[0]), int(jumps[0]), int(targets[0]), outputs)
        img_summary = ipb("img", imgs[None])
        writer.add_summary(img_summary, step)

        writer.add_summary(summaries, step)
        writer.flush()


if __name__ == '__main__':
    m = PrettyRRN()
    print(m.train_batch())
    print(m.val_batch())
