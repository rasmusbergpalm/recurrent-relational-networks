import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.data import Iterator

from model import Model
from tasks.go.data import *
import util


class AlphaGo(Model):
    n_blocks = 3  # 39
    n_hid = 32  # 256
    batch_size = 16  # 2048

    def __init__(self):
        self.is_training_ph = tf.placeholder(tf.bool, name='is_training')
        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.global_step = tf.Variable(0, trainable=False)
        regularizer = layers.l2_regularizer(1e-4)

        self.session = tf.Session()
        train, val, test = file_splits()
        train_iterator = self.iterator(train)
        valid_iterator = self.iterator(val)

        planes, winners, actions = tf.cond(
            self.is_training_ph,
            true_fn=lambda: train_iterator.get_next(),
            false_fn=lambda: valid_iterator.get_next()
        )

        def conv_bn_relu(x, n_filters, kernel_size):
            x = layers.conv2d(x, n_filters, kernel_size, activation_fn=None, weights_regularizer=regularizer)
            x = layers.batch_norm(x)
            return tf.nn.relu(x)

        def residual_block(x0):
            x = conv_bn_relu(x0, self.n_hid, 3)
            x = layers.conv2d(x, self.n_hid, 3, activation_fn=None, weights_regularizer=regularizer)
            x = layers.batch_norm(x)
            x = x + x0
            return tf.nn.relu(x)

        with tf.variable_scope('first'):
            x = conv_bn_relu(planes, self.n_hid, 3)

        for i in range(self.n_blocks):
            with tf.variable_scope('block/%02d' % i):
                x = residual_block(x)
        res_out = x

        with tf.variable_scope('head/policy'):
            x = conv_bn_relu(res_out, 2, 1)
            x = tf.reshape(x, (self.batch_size, 19 * 19 * 2))
            policy_logits = layers.fully_connected(x, 19 ** 2 + 1, activation_fn=None, weights_regularizer=regularizer)

        with tf.variable_scope('head/value'):
            x = conv_bn_relu(res_out, 1, 1)
            x = tf.reshape(x, (self.batch_size, 19 * 19))
            x = layers.fully_connected(x, self.n_hid, weights_regularizer=regularizer)
            value = tf.squeeze(layers.fully_connected(x, 1, activation_fn=tf.nn.tanh, weights_regularizer=regularizer))

        policy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=policy_logits))
        value_loss = tf.losses.mean_squared_error(winners, value)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = policy_loss + 0.01 * value_loss + reg_loss

        tf.summary.scalar('losses/policy', policy_loss)
        tf.summary.scalar('losses/value', value_loss)
        tf.summary.scalar('losses/reg', reg_loss)
        tf.summary.scalar('losses/total', self.loss)

        acc = tf.reduce_mean(tf.to_float(tf.equal(actions, tf.argmax(policy_logits, axis=1, output_type=tf.int32))))
        tf.summary.scalar('acc', acc)

        schedule = [
            (tf.greater(self.global_step, 700000), lambda: 0.00001),
            (tf.greater(self.global_step, 600000), lambda: 0.0001),
            (tf.greater(self.global_step, 400000), lambda: 0.001),
            (tf.greater(self.global_step, 200000), lambda: 0.01)
        ]

        lr = tf.case(
            schedule,
            default=lambda: 0.1,
            exclusive=True
        )
        tf.summary.scalar('lr', lr)

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, self.global_step)

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
        if step % 1000 == 0:
            self.train_writer.add_summary(summaries, step)
        return loss

    def val_batch(self):
        summaries, loss, step = self.session.run([self.summaries, self.loss, self.global_step], {self.is_training_ph: False})
        self.test_writer.add_summary(summaries, step)
        return loss

    def save(self, name):
        print("Saving %s..." % name)
        self.saver.save(self.session, name)

    def load(self, name):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)

    def iterator(self, files) -> Iterator:
        return tf.data.Dataset.from_generator(
            lambda: plane_encoded(positions(games(files))),
            (tf.float32, tf.float32, tf.int32),
            ((19, 19, 17), (), ())  # planes, winner, action
        ).repeat(-1).prefetch(100 * self.batch_size).shuffle(100 * self.batch_size).batch(self.batch_size).make_one_shot_iterator()


if __name__ == '__main__':
    m = AlphaGo()
    m.train_batch()
