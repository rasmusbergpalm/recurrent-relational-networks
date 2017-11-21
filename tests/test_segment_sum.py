from unittest import TestCase
import numpy as np

import tensorflow as tf


class TestSegmentSum(TestCase):
    def test_segment_sum(self):
        segment_ids = tf.placeholder(tf.int32, shape=(None,))  # [0, 0, 0, 1, 2, 2, 3, 3]
        data = tf.placeholder(tf.int32, shape=(None,))  # [5, 1, 7, 2, 3, 4, 1, 3]

        sum = tf.segment_sum(data, segment_ids)
        expected = np.array([13, 2, 7, 4])

        with tf.Session() as sess:
            actual = sess.run(sum, feed_dict={
                segment_ids: [0, 0, 0, 1, 2, 2, 3, 3],
                data: [5, 1, 7, 2, 3, 4, 1, 3]
            })
            self.assertTrue(np.allclose(expected, actual))
