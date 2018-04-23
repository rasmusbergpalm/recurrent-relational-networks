import os

import util
from model import Model
from tasks.diagnostics.pretty.data import PrettyClevr
from tensorboard.plugins.scalar.summary import pb as spb
import tensorflow as tf
import numpy as np
from tensorflow.python.data import Dataset
from tensorflow.contrib import layers


class PrettyMLP(Model):
    batch_size = 128
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    data = PrettyClevr()
    n_hidden = 256
    n_layers = 4

    def __init__(self):
        super().__init__()
        self.name = "%s %s" % (self.revision, self.message)
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.mode = tf.placeholder(tf.string, name='mode')
        self.is_training = tf.equal(self.mode, "train")
        self.saver = tf.train.Saver()

        train_iterator = self._iterator(self.data.train_generator, self.data.output_types(), self.data.output_shapes())
        dev_iterator = self._iterator(self.data.dev_generator, self.data.output_types(), self.data.output_shapes())
        test_iterator = self._iterator(self.data.test_generator, self.data.output_types(), self.data.output_shapes())

        print("Building graph...")
        self.org_img, positions, colors, markers, self.anchors, self.n_jumps, self.targets = tf.case(
            {
                tf.equal(self.mode, "train"): lambda: train_iterator.get_next(),
                tf.equal(self.mode, "dev"): lambda: dev_iterator.get_next(),
                tf.equal(self.mode, "test"): lambda: test_iterator.get_next(),
            },
            exclusive=True
        )
        bs = self.batch_size

        p1 = tf.expand_dims(positions, 1)  # (bs, 1, 8, 2)
        p2 = tf.expand_dims(positions, 2)  # (bs, 8, 1, 2)
        distances = tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), axis=3))  # (bs, 8,8)
        distances = tf.reshape(distances, (bs, 8 * 8))

        colors = tf.reshape(tf.one_hot(colors, 8), (bs, 8 * 8))  # (bs, 64)
        markers = tf.reshape(tf.one_hot(markers - 8, 8), (bs, 8 * 8))  # (bs, 64)
        positions = tf.reshape(positions, (bs, 8 * 2))  # (bs, 16)
        anchors = tf.one_hot(self.anchors, 16)  # (bs, 16)
        jumps = tf.one_hot(self.n_jumps, 8)  # (bs, 8)

        x = tf.concat([colors, markers, positions, distances, anchors, jumps], axis=1)  # (bs, 64+64+64+16+16+8)

        for i in range(self.n_layers):
            x = layers.fully_connected(x, num_outputs=self.n_hidden)
        x = layers.dropout(x, is_training=self.is_training)
        logits = layers.fully_connected(x, activation_fn=None, num_outputs=16)

        self.out = tf.argmax(logits, axis=1, output_type=tf.int32)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits) / tf.log(2.))
        tf.summary.scalar("loss", self.loss)

        acc = tf.reduce_mean(tf.to_float(tf.equal(self.out, self.targets)))
        tf.summary.scalar("acc", acc)

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        util.print_vars(tf.trainable_variables())

        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/pretty/%s/train/%s' % (self.revision, self.name), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/pretty/%s/test/%s' % (self.revision, self.name), self.session.graph)
        self.summaries = tf.summary.merge_all()

        self.session.run(tf.global_variables_initializer())

    def _iterator(self, generator, output_types, output_shapes):
        return Dataset.from_generator(
            generator,
            output_types,
            output_shapes
        ).batch(self.batch_size).prefetch(1).make_one_shot_iterator()

    def train_batch(self):
        step = self.session.run(self.global_step)
        if step % 1000 == 0:
            _, loss, summaries = self.session.run([self.train_op, self.loss, self.summaries], {self.mode: "train"})
            self.train_writer.add_summary(summaries, step)
        else:
            _, loss = self.session.run([self.train_op, self.loss], {self.mode: "train"})
        return loss

    def val_batch(self):
        loss, summaries, step, outputs, targets, jumps = self.session.run([self.loss, self.summaries, self.global_step, self.out, self.targets, self.n_jumps], {self.mode: "dev"})
        self._write_summaries(self.test_writer, summaries, jumps, targets, outputs, step)
        return loss

    def test_batches(self):
        print("Testing...")
        batches = []
        try:
            for i in range(1000):
                batches.append(self.session.run([self.org_img, self.anchors, self.n_jumps, self.targets, self.out], {self.mode: "dev"}))
        except tf.errors.OutOfRangeError:
            pass

        images, anchors, jumps, targets, outputs = zip(*batches)
        jumps = np.concatenate(jumps, axis=0)
        targets = np.concatenate(targets, axis=0)
        outputs = np.concatenate(outputs, axis=0)

        acc = np.array(self.compute_acc(jumps, outputs, targets))
        print(acc.shape)
        print(acc)
        # np.savez("results.npz", images=images, anchors=anchors, jumps=jumps, targets=targets, outputs=outputs, acc=acc)

    def _write_summaries(self, writer, summaries, jumps, targets, outputs, step):
        accs = self.compute_acc(jumps, outputs, targets)
        for jump in range(8):
            writer.add_summary(spb("acc/0/%d" % jump, accs[jump]), step)
        writer.add_summary(summaries, step)
        writer.flush()

    def compute_acc(self, jumps, outputs, targets):
        """
        :param jumps: (bs, )
        :param outputs: (bs,)
        :param targets: (bs,)
        :return:
        """
        jumps_acc = []
        equal = outputs == targets
        for i in range(8):
            jumps_i = jumps == i
            if any(jumps_i):
                acc = np.mean(equal[jumps_i])
                jumps_acc.append(acc)
        return jumps_acc

    def save(self, name):
        self.saver.save(self.session, name)

    def load(self, name):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)


if __name__ == '__main__':
    m = PrettyMLP()
    print(m.val_batch())
    print(m.train_batch())
