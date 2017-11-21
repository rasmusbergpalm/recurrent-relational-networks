from unittest import TestCase

import numpy as np
import tensorflow as tf

from message_passing import message_passing


class TestMessagePassing(TestCase):
    def test_message_passing(self):
        x = np.random.randn(3, 2).astype(np.float32)
        nodes = tf.constant(x, tf.float32, name='nodes')
        edges = tf.constant(np.array([[0, 1], [1, 2], [2, 1]]), tf.int32)
        edge_features = tf.zeros((3, 1), tf.float32)

        def message_fn(x):
            return x[:, 0:2] + x[:, 2:4]

        out = message_passing(nodes, edges, edge_features, message_fn, 1.0)
        expected = np.array([
            [0, 0],  # no messages for node 0
            (x[0] + x[1]) + (x[2] + x[1]),  # 0 to 1, and 2 to 1
            (x[1] + x[2])  # 1 to 2
        ], dtype=np.float32)

        with tf.Session().as_default():
            self.assertTrue(np.allclose(expected, out.eval()))
