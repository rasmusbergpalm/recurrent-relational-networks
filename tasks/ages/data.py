import random

import networkx as nx
import tensorflow as tf


class Ages:
    n = 8

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

    def generator(self):
        while True:
            g = nx.random_tree(self.n)

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

            yield sources, targets, types, diffs, question, answer, n_jumps
