import os

import matplotlib
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorboard.plugins.image.summary import pb as ipb
from tensorboard.plugins.scalar.summary import pb as spb
from tensorflow.contrib import layers
from tensorflow.python.data import Dataset

import util
from message_passing import message_passing
from model import Model
from tasks.diagnostics.pretty.data import PrettyClevr, fig2array

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PrettyRRN(Model):
    number = 1
    batch_size = 100
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    n_objects = 8
    data = PrettyClevr()
    n_steps = 1
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

        regularizer = layers.l2_regularizer(0.)

        train_iterator = self._iterator(self.data.train_generator, self.data.output_types(), self.data.output_shapes())
        dev_iterator = self._iterator(self.data.dev_generator, self.data.output_types(), self.data.output_shapes())
        test_iterator = self._iterator(self.data.test_generator, self.data.output_types(), self.data.output_shapes())

        n_nodes = 8
        n_anchors_targets = len(self.data.i2s)

        def mlp(x, scope, n_hid=self.n_hidden, n_out=self.n_hidden, keep_prob=1.0):
            with tf.variable_scope(scope):
                for i in range(1):
                    x = layers.fully_connected(x, n_hid, weights_regularizer=regularizer)
                x = layers.dropout(x, keep_prob=keep_prob, is_training=self.is_training)
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

            colors = tf.reshape(tf.one_hot(colors, 8), (bs * n_nodes, 8))
            markers = tf.reshape(tf.one_hot(markers - 8, 8), (bs * n_nodes, 8))
            positions = tf.reshape(positions, (bs * n_nodes, 2))

            distances = tf.gather(positions, edges)  # (n_edges, 2, 2)
            distances = tf.sqrt(tf.reduce_sum(tf.square(distances[:, 0] - distances[:, 1]), axis=1, keep_dims=True))  # (n_edges, 1)

            question = tf.concat([tf.one_hot(anchors, n_anchors_targets), tf.one_hot(n_jumps, self.n_objects)], axis=1)
            question = tf.gather(question, segment_ids)

            x = tf.concat([positions, colors, markers, question], axis=1)
            x = mlp(x, 'pre')

            edge_features = distances

            with tf.variable_scope('steps'):
                outputs = []
                losses = []
                x0 = x

                for step in range(self.n_steps):
                    x = message_passing(x, edges, edge_features, lambda x: mlp(x, 'message-fn'))
                    x = mlp(tf.concat([x, x0], axis=1), 'post')

                    logits = tf.unsorted_segment_sum(x, segment_ids, bs)
                    logits = mlp(logits, "out", n_out=n_anchors_targets, keep_prob=0.5)

                    out = tf.argmax(logits, axis=1)
                    outputs.append(out)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits) / tf.log(2.))
                    losses.append(loss)

                    tf.get_variable_scope().reuse_variables()

            return losses, outputs

        self.org_img, positions, colors, markers, self.anchors, self.n_jumps, self.targets = tf.case(
            {
                tf.equal(self.mode, "train"): lambda: train_iterator.get_next(),
                tf.equal(self.mode, "dev"): lambda: dev_iterator.get_next(),
                tf.equal(self.mode, "test"): lambda: test_iterator.get_next(),
            },
            exclusive=True
        )

        log_losses, outputs = util.batch_parallel(forward, self.devices, img=self.org_img, anchors=self.anchors, n_jumps=self.n_jumps, targets=self.targets, positions=positions, colors=colors, markers=markers)
        log_losses = tf.reduce_mean(log_losses)
        self.outputs = tf.concat(outputs, axis=1)  # (splits, steps, bs)

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('reg_loss', reg_loss)

        self.loss = tf.reduce_mean(log_losses) + reg_loss
        tf.summary.scalar('loss', self.loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)
            self.train_step = self.optimizer.apply_gradients(gvs, global_step=self.global_step)

        for i, (g, v) in enumerate(gvs):
            tf.summary.histogram("grads/%03d/%s" % (i, v.name), g)
            tf.summary.histogram("vars/%03d/%s" % (i, v.name), v)
            tf.summary.histogram("g_ratio/%03d/%s" % (i, v.name), tf.log(tf.abs(g) + 1e-8) - tf.log(tf.abs(v) + 1e-8))

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
            _, loss, summaries = self.session.run([self.train_step, self.loss, self.summaries], {self.mode: "train"})
            self.train_writer.add_summary(summaries, step)
        else:
            _, loss = self.session.run([self.train_step, self.loss], {self.mode: "train"})
        return loss

    def val_batch(self):
        loss, summaries, step, img, anchors, jumps, targets, outputs = self.session.run([self.loss, self.summaries, self.global_step, self.org_img, self.anchors, self.n_jumps, self.targets, self.outputs], {self.mode: "dev"})
        self._write_summaries(self.test_writer, summaries, img, anchors, jumps, targets, outputs, step)
        return loss

    def test_batches(self):
        print("Testing...")
        batches = []
        try:
            while True:
                batches.append(self.session.run([self.org_img, self.anchors, self.n_jumps, self.targets, self.outputs], {self.mode: "test"}))
        except tf.errors.OutOfRangeError:
            pass

        images, anchors, jumps, targets, outputs = zip(*batches)
        jumps = np.concatenate(jumps, axis=0)
        targets = np.concatenate(targets, axis=0)
        outputs = np.concatenate(outputs, axis=1)

        acc = np.array(self.compute_acc(jumps, outputs, targets))
        print(acc.shape)
        print(acc)
        np.savez("results.npz", images=images, anchors=anchors, jumps=jumps, targets=targets, outputs=outputs, acc=acc)

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
        accs = self.compute_acc(jumps, outputs, targets)
        for step in range(self.n_steps):
            for jump in range(8):
                writer.add_summary(spb("acc/%d/%d" % (step, jump), accs[step][jump]), step)

        imgs = self._render(img[0], int(anchors[0]), int(jumps[0]), int(targets[0]), outputs)
        img_summary = ipb("img", imgs[None])
        writer.add_summary(img_summary, step)

        writer.add_summary(summaries, step)
        writer.flush()

    def compute_acc(self, jumps, outputs, targets):
        accs = []
        for t in range(self.n_steps):
            jumps_acc = []
            equal = outputs[t] == targets
            for i in range(8):
                jumps_i = jumps == i
                if any(jumps_i):
                    acc = np.mean(equal[jumps_i])
                    jumps_acc.append(acc)
            accs.append(jumps_acc)
        return accs


if __name__ == '__main__':
    m = PrettyRRN()
    print(m.val_batch())
    print(m.train_batch())
