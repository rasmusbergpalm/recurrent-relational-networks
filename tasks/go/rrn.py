import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import LSTMCell

import util
from message_passing import message_passing
from model import Model
from tasks.go.data import file_splits, graph_encoded, positions, games


class GoRecurrentRelationalNet(Model):
    devices = util.get_devices()
    batch_size = (256 // len(devices)) * len(devices)
    emb_size = 16
    n_steps = 32
    n_hidden = 64
    size = 19

    def __init__(self):
        self.is_training_ph = tf.placeholder(tf.bool)
        train, val, test = file_splits()
        train_iterator = self.iterator(train)
        valid_iterator = self.iterator(val)

        print("Building graph...")
        self.session = tf.Session()
        regularizer = layers.l2_regularizer(1e-4)
        edges = self.edges()
        edge_indices = tf.constant([(i + (b * self.size ** 2), j + (b * self.size ** 2)) for b in range(self.batch_size // len(self.devices)) for i, j in edges], tf.int32)
        n_edges = tf.shape(edge_indices)[0]
        edge_features = tf.zeros((n_edges, 1), tf.float32)
        positions = tf.constant([[(i, j) for i in range(self.size) for j in range(self.size)] for b in range(self.batch_size // len(self.devices))], tf.int32)  # (bs, 361, 2)
        rows = layers.embed_sequence(positions[:, :, 0], self.size, self.emb_size, scope='row-embeddings')  # bs, 361, emb_size
        cols = layers.embed_sequence(positions[:, :, 1], self.size, self.emb_size, scope='cols-embeddings')  # bs, 361, emb_size

        def mlp(x, scope):
            with tf.variable_scope(scope):
                for i in range(3):
                    x = layers.fully_connected(x, self.n_hidden, weights_regularizer=regularizer)
                return layers.fully_connected(x, self.n_hidden, weights_regularizer=regularizer, activation_fn=None)

        def forward(stones, player, winners, actions):
            n_player_embedding = 4
            x = layers.embed_sequence(stones, 3, self.emb_size, scope='stone-embedding')  # bs, 361, emb_size
            player_emb = layers.embed_sequence(player, 2, embed_dim=n_player_embedding, scope='player-embedding')
            x = tf.concat([x, rows, cols, player_emb], axis=2)
            x = tf.reshape(x, (-1, 3 * self.emb_size + n_player_embedding))
            x = mlp(x, 'pre-fn')
            x0 = x

            policy_loss, value_loss, acc = [], [], []
            lstm_cell = LSTMCell(self.n_hidden)
            state = lstm_cell.zero_state(tf.shape(x)[0], tf.float32)
            with tf.variable_scope('steps'):
                for step in range(self.n_steps):
                    x = message_passing(x, edge_indices, edge_features, lambda x: mlp(x, 'message-fn'))
                    x = mlp(tf.concat([x, x0], axis=1), 'post-fn')
                    x, state = lstm_cell(x, state)

                    value = tf.nn.tanh(tf.reduce_sum(tf.reshape(layers.fully_connected(x, num_outputs=1, activation_fn=None, scope='value'), (-1, self.size ** 2)), axis=1))
                    value_loss.append(tf.reduce_mean(tf.square(winners - value)))

                    policy_logits = tf.reshape(layers.fully_connected(x, num_outputs=1, activation_fn=None, scope='policy'), (-1, self.size ** 2))
                    policy_logits = tf.concat([policy_logits, tf.zeros((tf.shape(policy_logits)[0], 1))], axis=1)
                    policy_loss.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=policy_logits)))
                    acc.append(tf.reduce_mean(tf.to_float(tf.equal(actions, tf.argmax(policy_logits, axis=1, output_type=tf.int32)))))
                    tf.get_variable_scope().reuse_variables()

            return policy_loss, value_loss, acc

        stones, player, winners, actions = tf.cond(
            self.is_training_ph,
            true_fn=lambda: train_iterator.get_next(),
            false_fn=lambda: valid_iterator.get_next()
        )

        def avg_n(x):
            return tf.reduce_mean(tf.stack(x, axis=0), axis=0)

        policy_loss, value_loss, acc = util.batch_parallel(forward, avg_n, self.devices, stones=stones, player=player, winners=winners, actions=actions)  # n_steps

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(policy_loss) + 0.01 * tf.reduce_mean(value_loss) + reg_loss

        tf.summary.scalar('losses/total', self.loss)
        tf.summary.scalar('losses/reg', reg_loss)
        for step in range(self.n_steps):
            tf.summary.scalar('steps/%d/acc' % step, acc[step])
            tf.summary.scalar('steps/%d/losses/policy' % step, policy_loss[step])
            tf.summary.scalar('steps/%d/losses/value' % step, value_loss[step])

        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step, colocate_gradients_with_ops=True)

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        util.print_vars(tf.trainable_variables())

        revision = os.environ.get('REVISION')
        message = os.environ.get('MESSAGE')
        name = "%s %s" % (revision, message)
        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/go/%s/train/%s' % (revision, name), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/go/%s/test/%s' % (revision, name), self.session.graph)
        self.summaries = tf.summary.merge_all()

    def edges(self):
        edges = []
        idx = np.arange(self.size ** 2).reshape(self.size, self.size)
        for r in range(self.size):
            for c in range(self.size):
                if r + 1 < self.size:
                    edges.append((idx[r, c], idx[r + 1, c]))
                if c + 1 < self.size:
                    edges.append((idx[r, c], idx[r, c + 1]))
        edges += [(j, i) for i, j in edges]
        return edges

    def iterator(self, files):
        return tf.data.Dataset.from_generator(
            lambda: graph_encoded(positions(games(files))),
            (tf.int32, tf.int32, tf.float32, tf.int32),
            ((19 ** 2,), (19 ** 2,), (), ())  # stones, color, winner, action
        ).repeat(-1).prefetch(100 * self.batch_size).shuffle(100 * self.batch_size).batch(self.batch_size).make_one_shot_iterator()

    def save(self, name):
        print("Saving %s..." % name)
        self.saver.save(self.session, name)

    def load(self, name):
        print("Loading %s..." % name)
        self.saver.restore(self.session, name)

    def train_batch(self):
        _, loss, summaries, step = self.session.run([self.train_step, self.loss, self.summaries, self.global_step], {self.is_training_ph: True})
        if step % 1000 == 0:
            self.train_writer.add_summary(summaries, step)

        return loss

    def val_batch(self):
        loss, summaries, step = self.session.run([self.loss, self.summaries, self.global_step], {self.is_training_ph: False})
        self.test_writer.add_summary(summaries, step)
        return loss


if __name__ == '__main__':
    m = GoRecurrentRelationalNet()
    for i in range(10):
        m.train_batch()
