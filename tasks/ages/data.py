import random

import networkx as nx
import tensorflow as tf
from itertools import product


class Ages:
    n = 8

    def __init__(self):
        print("Pre-generating graphs...")
        graphs = [nx.from_prufer_sequence(seq) for seq in product(range(self.n), repeat=self.n - 2)]
        random.seed(0)
        graphs = random.sample(graphs, len(graphs))
        n_train = round(len(graphs) * 0.9)
        self.train_graphs = graphs[:n_train]
        self.test_graphs = graphs[n_train:]
        print("Done.")

    def types_and_shapes(self):
        return zip(*(
            (tf.int32, (self.n,)),  # sources,
            (tf.int32, (self.n,)),  # targets,
            (tf.int32, (self.n,)),  # types
            (tf.int32, (self.n,)),  # diffs,
            (tf.int32, ()),  # question,
            (tf.int32, ()),  # answer,
            (tf.int32, ()),  # n_jumps
        ))

    def test_generator(self):
        while True:
            yield self.encode_sequence(random.choice(self.test_graphs))

    def train_generator(self):
        while True:
            yield self.encode_sequence(random.choice(self.train_graphs))

    def encode_sequence(self, g):
        ages = random.sample(range(0, 99), self.n)
        anchor = random.randint(0, self.n - 1)
        edges = list(g.edges)

        sources, targets = list(zip(*list(g.edges)))
        types = [0 if ages[i] >= ages[j] else 1 for i, j in edges]
        diffs = [abs(ages[i] - ages[j]) for i, j in edges]

        # anchor fact
        sources += (anchor,)
        targets += (anchor,)
        types += [2]
        diffs += [ages[anchor]]

        question = random.randint(0, self.n - 1)
        n_jumps = len(nx.shortest_path(g, anchor, question)) - 1
        answer = ages[question]

        return sources, targets, types, diffs, question, answer, n_jumps
