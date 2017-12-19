import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.data import Iterator

from model import Model
from tasks.go.data import *
import util
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell


class RNN(Model):
    n_layers = 2
    n_hid = 128
    devices = util.get_devices()
    batch_size = 16 // len(devices) * len(devices)
    size = 19

    def __init__(self):
        self.is_training_ph = tf.placeholder(tf.bool, name='is_training')
        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.global_step = tf.Variable(0, trainable=False)

        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        train, val, test = file_splits()
        train_iterator = self.iterator(train)
        valid_iterator = self.iterator(val)

        def forward(states, moves, values):
            mask = tf.to_float(tf.not_equal(values, -9.))
            n_mask = tf.reduce_sum(mask)

            multi_cell = MultiRNNCell([LSTMCell(self.n_hid) for i in range(self.n_layers)])
            outputs, _ = tf.nn.dynamic_rnn(multi_cell, states, initial_state=multi_cell.zero_state(self.batch_size // len(self.devices), tf.float32))

            policy_logits = layers.fully_connected(outputs, self.size ** 2 + 1, activation_fn=None)
            winners = layers.fully_connected(outputs, 1, activation_fn=tf.nn.tanh)

            policy_loss = tf.reduce_sum(mask * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=moves, logits=policy_logits)) / n_mask
            value_loss = tf.reduce_sum(mask * tf.square(winners[:, :, 0] - values)) / n_mask
            acc = tf.reduce_mean(tf.to_float(tf.equal(moves, tf.argmax(policy_logits, axis=2, output_type=tf.int32))))

            return policy_loss, value_loss, acc

        states, moves, values = tf.cond(
            self.is_training_ph,
            true_fn=lambda: train_iterator.get_next(),
            false_fn=lambda: valid_iterator.get_next()
        )

        policy_loss, value_loss, acc = util.batch_parallel(forward, tf.reduce_mean, self.devices, states=states, moves=moves, values=values)

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = policy_loss + 0.01 * value_loss + reg_loss

        tf.summary.scalar('losses/total', self.loss)
        tf.summary.scalar('losses/policy', policy_loss)
        tf.summary.scalar('losses/value', value_loss)
        tf.summary.scalar('losses/reg', reg_loss)
        tf.summary.scalar('acc', acc)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)
        self.train_op = self.optimizer.minimize(self.loss, self.global_step, colocate_gradients_with_ops=True)

        revision = os.environ.get('REVISION')
        message = os.environ.get('MESSAGE')
        name = "%s %s" % (revision, message)
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/go/%s/train/%s' % (revision, name), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/go/%s/test/%s' % (revision, name), self.session.graph)
        self.summaries = tf.summary.merge_all()
        util.print_vars(tf.trainable_variables())
        self.session.run(tf.global_variables_initializer())

    def train_batch(self):
        _, summaries, loss, step = self.session.run([self.train_op, self.summaries, self.loss, self.global_step], {self.is_training_ph: True})
        if step % 10 == 0:
            self.train_writer.add_summary(summaries, step)
            self.train_writer.flush()
        return loss

    def val_batch(self):
        summaries, loss, step = self.session.run([self.summaries, self.loss, self.global_step], {self.is_training_ph: False})
        self.test_writer.add_summary(summaries, step)
        self.test_writer.flush()
        return loss

    def save(self, name):
        print("Saving %s..." % name)
        self.saver.save(self.session, name)

    def load(self, name):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)

    def iterator(self, files) -> Iterator:
        return tf.data.Dataset.from_generator(
            lambda: rnn_encoded(games(files)),
            (tf.float32, tf.int32, tf.float32)  # states, moves, values
        ).repeat(-1).shuffle(100 * self.batch_size).padded_batch(self.batch_size, padded_shapes=((None, 19 ** 2), (None,), (None,)), padding_values=(0., 0, -9.)).prefetch(3).make_one_shot_iterator()


if __name__ == '__main__':
    m = RNN()
    m.train_batch()
