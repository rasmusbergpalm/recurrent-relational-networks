from unittest import TestCase

from keras.preprocessing.sequence import pad_sequences

from tasks.babi.rrn import BaBiRecurrentRelationalNet
import numpy as np


class TestBAbIReasonNet(TestCase):

    def test_get_placeholders(self):
        model = BaBiRecurrentRelationalNet(False)
        batch = [
            {
                'facts': [
                    [1, 2, 3],
                    [4, 5, 6, 7]
                ],
                'q': [8, 9],
                'a': 10,
                'fact_positions': [0, 1],
                'task_idx': 0
            },
            {
                'facts': [
                    [11, 12, 13],
                    [14, 15, 16, 17],
                    [18, 19]
                ],
                'q': [20, 21, 22],
                'a': 23,
                'fact_positions': [0, 1, 2],
                'task_idx': 0
            }
        ]

        ph = model.get_feed_dict(model.encode_batch(batch, True))

        expected_facts = pad_sequences([
            [1, 2, 3],
            [4, 5, 6, 7],
            [11, 12, 13],
            [14, 15, 16, 17],
            [18, 19]
        ], padding='post')
        self.assertTrue(np.allclose(expected_facts, ph[model.facts_ph]))

        expected_f_seq_length = [3, 4, 3, 4, 2]
        self.assertListEqual(expected_f_seq_length, ph[model.f_seq_length_ph].tolist())

        expected_questions = pad_sequences([
            [8, 9],
            [20, 21, 22]
        ], padding='post')
        self.assertTrue(np.allclose(expected_questions, ph[model.question_ph]))

        expected_q_seq_length = [2, 3]
        self.assertListEqual(expected_q_seq_length, ph[model.q_seq_length_ph].tolist())

        expected_answers = [10, 23]
        self.assertListEqual(expected_answers, ph[model.answers_ph].tolist())

        expected_segments = [0, 0, 1, 1, 1]
        self.assertListEqual(expected_segments, ph[model.fact_segments_ph].tolist())

        expected_edge_indices = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],

            [2, 2],
            [2, 3],
            [2, 4],
            [3, 2],
            [3, 3],
            [3, 4],
            [4, 2],
            [4, 3],
            [4, 4]
        ]
        self.assertListEqual(expected_edge_indices, ph[model.edge_indices_ph].tolist())

        expected_edge_segments = [0] * 4 + [1] * 9
        self.assertListEqual(expected_edge_segments, ph[model.edge_segments_ph].tolist())

        expected_fact_positions = [0, 1, 0, 1, 2]
        self.assertListEqual(expected_fact_positions, ph[model.facts_pos_ph].tolist())
