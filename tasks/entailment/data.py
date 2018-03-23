import os
import random
import urllib.request
import zipfile

from tasks.entailment.parser import Parser, propositional_language, ParseResult


class Data:
    url = "https://www.dropbox.com/s/ufpau116axq92iz/entailment.zip?dl=1"
    base_dir = (os.environ.get('DATA_DIR') or "/tmp")
    zip_fname = base_dir + "/entailment.zip"
    dest_dir = base_dir + '/entailment/'

    def __init__(self):
        self.language = propositional_language()
        self.dict = {k: i for i, k in enumerate(self.language.symbols)}
        self.parser = Parser(self.language)
        if not os.path.exists(self.dest_dir):
            print("Downloading data...")

            urllib.request.urlretrieve(self.url, self.zip_fname)
            with zipfile.ZipFile(self.zip_fname) as f:
                f.extractall(self.base_dir)

    def sample_generator(self, fname):
        with open(self.dest_dir + '/' + fname, 'r') as fp:
            samples = [line.strip().split(',') for line in fp.readlines()]
        while True:
            for a, b, t, h1, h2, h3 in random.sample(samples, len(samples)):
                a = self.parser.parse(a)
                b = self.parser.parse(b)
                a, b = self.normalize(a, b)

                n_a, e_a = self.encode(a)
                n_b, e_b = self.encode(b)
                yield n_a, e_a, n_b, e_b, int(t)

    def encode(self, p: ParseResult):
        edges = []
        for i, inputs in enumerate(p.inputs):
            for j in inputs:
                edges.append((i + j, i))
        return [self.dict[op] for op in p.ops], edges

    def normalize(self, a: ParseResult, b: ParseResult):
        p = {op for op in a.ops + b.ops if op in self.language.predicates}
        perm = {a: b for a, b in zip(random.sample(p, len(p)), self.language.predicates[:len(p)])}
        a = ParseResult(a.expression, [perm[op] if op in perm else op for op in a.ops], a.inputs)
        b = ParseResult(b.expression, [perm[op] if op in perm else op for op in b.ops], b.inputs)
        return a, b


if __name__ == '__main__':
    d = Data()
    for n1, e1, n2, e2, t in d.sample_generator('train.txt'):
        i = 0
