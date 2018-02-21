import os

import matplotlib
import tensorflow as tf
from tensorboard.plugins.image.summary import pb as ipb
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
    batch_size = 32
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    n = 8
    data = PrettyClevr()
    n_steps = 1
    n_hidden = 32

    def __init__(self):
        super().__init__()
        self.name = "%s %s" % (self.revision, self.message)

        print("Building graph...")
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-4)

        iterator = self._iterator(self.data)

        self.org_img, self.anchors, self.n_jumps, self.targets = iterator.get_next()
        self.img = ((1. - tf.to_float(self.org_img) / 255.) - 0.5)  # (bs, h, w, 3)

        # self.xy = tf.tile(tf.expand_dims(tf.transpose(tf.meshgrid(tf.linspace(0., 1., 128), tf.linspace(0., 1., 128)), (1, 2, 0)), axis=0), (self.batch_size, 1, 1, 1))

        def mlp(x, scope, n_hid=self.n_hidden, n_out=self.n_hidden):
            with tf.variable_scope(scope):
                for i in range(3):
                    x = layers.fully_connected(x, n_hid)
                return layers.fully_connected(x, n_out, activation_fn=None)

        # self.n = 8 * 8
        # n_nodes = self.batch_size * self.n
        # x = tf.reshape(x, (n_nodes, self.n_hidden))

        # edges = [(i, j) for i in range(self.n) for j in range(self.n)]
        # edges = tf.constant([(i + (b * self.n), j + (b * self.n)) for b in range(self.batch_size) for i, j in edges], tf.int32)
        # n_edges = tf.shape(edges)[0]

        n_anchors_targets = len(self.data.i2s)
        q = tf.concat([tf.one_hot(self.anchors, n_anchors_targets), tf.one_hot(self.n_jumps, self.n)], axis=1)  # (bs, 24)
        q = mlp(q, "q")
        q = tf.reshape(q, (self.batch_size, 1, 1, self.n_hidden))
        q = tf.tile(q, (1, 128, 128, 1))

        # x = tf.concat([self.img, self.xy], axis=-1)
        x = tf.concat([self.img, q], axis=-1)
        with tf.variable_scope('encoder'):
            for i in range(7):
                x = layers.conv2d(x, num_outputs=self.n_hidden, kernel_size=11, stride=1)  # (bs, h, w, 128)
                x = layers.max_pool2d(x, 2, 2)

        x = tf.reshape(x, (self.batch_size, self.n_hidden))
        logits = layers.fully_connected(x, n_anchors_targets, activation_fn=None)
        # x = tf.concat([x, q], axis=1)
        # logits = mlp(x, "out", n_hid=self.n_hidden, n_out=n_anchors_targets)

        # edge_features = tf.reshape(tf.tile(tf.expand_dims(question, 1), [1, self.n ** 2, 1]), [n_edges, n_anchors_targets + self.n])

        # n_nodes = self.n * self.batch_size

        with tf.variable_scope('steps'):
            self.outputs = []
            losses = []
            # x0 = x
            # lstm_cell = LSTMCell(self.n_hidden)
            # state = lstm_cell.zero_state(n_nodes, tf.float32)
            for step in range(self.n_steps):
                # x = message_passing(x, edges, edge_features, lambda x: mlp(x, 'message-fn'))
                # x = mlp(tf.concat([x, x0], axis=1), 'post')
                # tf.summary.histogram("activations/%d" % step, x)
                # x = layers.batch_norm(x, scope='bn')
                # x, state = lstm_cell(x, state)

                # logits = mlp(x, 'logits', n_anchors_targets)
                # logits = tf.reshape(logits, (self.batch_size, self.n, n_anchors_targets))
                # logits = tf.reduce_sum(logits, axis=1)  # (bs, n_anchors_targets)

                out = tf.argmax(logits, axis=1)
                self.outputs.append(out)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits) / tf.log(2.))
                losses.append(loss)

                tf.summary.scalar('steps/%d/loss' % step, loss)
                tf.get_variable_scope().reuse_variables()

        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', self.loss)

        gvs = self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)
        for g, v in gvs:
            tf.summary.histogram("grads/" + v.name, g)
            tf.summary.histogram("vars/" + v.name, v)
            tf.summary.histogram("g_ratio/" + v.name, g / (v + 1e-8))

        # gvs = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in gvs]
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
        _, loss = self.session.run([self.train_step, self.loss])
        return loss

    def val_batch(self):
        loss, summaries, step, img, anchors, jumps, targets, outputs = self.session.run([self.loss, self.summaries, self.global_step, self.org_img, self.anchors, self.n_jumps, self.targets, self.outputs])
        self._write_summaries(self.test_writer, summaries, img, anchors, jumps, targets, outputs, step)
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

    def _render(self, img, anchor, jump, target, outputs):
        fig = plt.figure(figsize=(2.56, 2.56), frameon=False)
        plt.imshow(img)
        out_str = str([self.data.i2s[str(output[0])] for output in outputs])
        plt.title("%s %d %s %s" % (self.data.i2s[str(anchor)], jump, self.data.i2s[str(target)], out_str))
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        return fig2array(fig)

    def _write_summaries(self, writer, summaries, img, anchors, jumps, targets, outputs, step):
        imgs = self._render(img[0], int(anchors[0]), int(jumps[0]), int(targets[0]), outputs)
        img_summary = ipb("img", imgs[None])
        writer.add_summary(img_summary, step)

        writer.add_summary(summaries, step)
        writer.flush()


if __name__ == '__main__':
    m = PrettyRRN()
    print(m.train_batch())
    print(m.val_batch())
