import os

import matplotlib
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.data import Dataset

import util
from message_passing import message_passing
from model import Model
from tasks.entailment.data import Entailment

matplotlib.use('Agg')


class EntailmentRRN(Model):
    number = 1
    batch_size = 128
    data = Entailment()
    n_steps = 16
    n_hidden = 128
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')

    def __init__(self):
        super().__init__()
        self.name = "%s %s" % (self.revision, self.message)

        print("Building graph...")
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.is_training_ph = tf.placeholder(bool, name='is_training')
        regularizer = layers.l2_regularizer(0.)

        train_iterator = self._iterator(lambda: self.data.batch_generator('train.txt', self.batch_size), self.data.output_types(), self.data.output_shapes())
        dev_iterator = self._iterator(lambda: self.data.batch_generator('validate.txt', self.batch_size), self.data.output_types(), self.data.output_shapes())

        def mlp(x, scope, n_layers=1, n_hid=self.n_hidden, n_out=self.n_hidden, keep_prob=1.0):
            with tf.variable_scope(scope):
                for i in range(n_layers):
                    x = layers.fully_connected(x, n_hid, weights_regularizer=regularizer)
                x = layers.dropout(x, keep_prob=keep_prob, is_training=self.is_training_ph)
                return layers.fully_connected(x, n_out, weights_regularizer=regularizer, activation_fn=None)

        nodes_a, edges_a, segments_a, heads_a, nodes_b, edges_b, segments_b, heads_b, targets = tf.cond(
            self.is_training_ph,
            true_fn=lambda: train_iterator.get_next(),
            false_fn=lambda: dev_iterator.get_next(),
        )

        def encode_graph(nodes, edges, segments, heads, reuse):
            edge_features = tf.zeros((tf.shape(edges)[0], 1))
            with tf.variable_scope('encoder', reuse=reuse):
                x = layers.embed_sequence(nodes, vocab_size=len(self.data.dict), embed_dim=self.n_hidden)
                x0 = x

                with tf.variable_scope('steps'):
                    for step in range(self.n_steps):
                        x = message_passing(x, edges, edge_features, lambda x: mlp(x, 'message-fn'))
                        x = mlp(tf.concat([x, x0], axis=1), 'post')
                        tf.get_variable_scope().reuse_variables()

                return tf.gather(x, heads)

        ea = encode_graph(nodes_a, edges_a, segments_a, heads_a, False)
        eb = encode_graph(nodes_b, edges_b, segments_b, heads_b, True)
        logits = mlp(tf.concat([ea, eb], axis=1), "logits", n_out=1)

        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(logits, 0)), targets)))
        tf.summary.scalar('acc', acc)

        log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(targets, (-1, 1)), logits=logits) / tf.log(2.)

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('reg_loss', reg_loss)

        self.loss = tf.reduce_mean(log_loss) + reg_loss
        tf.summary.scalar('loss', self.loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)
            self.train_step = self.optimizer.apply_gradients(gvs, global_step=self.global_step)

        for g, v in gvs:
            tf.summary.histogram("grads/" + v.name, g)
            tf.summary.histogram("vars/" + v.name, v)
            tf.summary.histogram("g_ratio/" + v.name, tf.log(g) - tf.log(v + 1e-8))

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        util.print_vars(tf.trainable_variables())

        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/entail/%s/train/%s' % (self.revision, self.name), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/entail/%s/test/%s' % (self.revision, self.name), self.session.graph)
        self.summaries = tf.summary.merge_all()

    def train_batch(self):
        step = self.session.run(self.global_step)
        if step % 1000 == 0:
            _, loss, summaries = self.session.run([self.train_step, self.loss, self.summaries], {self.is_training_ph: True})
            self.train_writer.add_summary(summaries, step)
            self.train_writer.flush()
        else:
            _, loss = self.session.run([self.train_step, self.loss], {self.is_training_ph: True})
        return loss

    def val_batch(self):
        loss, summaries, step = self.session.run([self.loss, self.summaries, self.global_step], {self.is_training_ph: False})
        self.test_writer.add_summary(summaries, step)
        self.test_writer.flush()
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
        ).prefetch(1).make_one_shot_iterator()


if __name__ == '__main__':
    m = EntailmentRRN()
    print(m.train_batch())
    print(m.val_batch())
