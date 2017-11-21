from unittest import TestCase
import numpy as np

import tensorflow as tf


class TestScatter(TestCase):
    def test_scatter_nd(self):
        indices = tf.constant([[4], [3], [1], [7], [1]])
        updates = tf.constant([9, 10, 11, 12, 13])
        shape = tf.constant([8])
        scatter = tf.scatter_nd(indices, updates, shape)
        expected = np.array([0, 24, 0, 10, 9, 0, 0, 12])
        for i in range(1000):
            with tf.Session() as sess:
                actual = sess.run(scatter)
                self.assertTrue(np.allclose(expected, actual))

    def test_scatter_nd_add(self):
        indices = tf.constant([[4], [3], [1], [7], [1]])
        updates = tf.constant([9, 10, 11, 12, 13])
        shape = tf.constant([8])
        ref = tf.Variable(tf.zeros(shape, dtype=tf.int32), trainable=False)
        scatter = tf.scatter_nd_add(ref, indices, updates)
        expected = np.array([0, 24, 0, 10, 9, 0, 0, 12])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(scatter)
            self.assertTrue(np.allclose(expected, actual))
