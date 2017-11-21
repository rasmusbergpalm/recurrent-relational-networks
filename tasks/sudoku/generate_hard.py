import random
import urllib.request

# The original 49.151 sudoku's with 17 givens are from Gordon Royle (http://staffhome.ecm.uwa.edu.au/~00013890/sudokumin.php)
# I've found all the solutions using Norvigs code (http://norvig.com/sudoku.html) and uploaded the file with puzzles and solutions in one file

url = "https://www.dropbox.com/s/bcf5q30oryg0csw/sudoku17.txt?dl=1"
fname = "/tmp/sudoku17.txt"
urllib.request.urlretrieve(url, fname)

with open(fname) as f:
    lines = f.readlines()

hard = [line.strip().split(',') for line in lines]
hard = random.sample(hard, len(hard))  # shuffled copy

n_test = 10000
n_valid = 1000

test_pool = hard[:n_test]
valid_pool = hard[n_test:n_test + n_valid]
train_pool = hard[n_test + n_valid:]


def permute(sample):
    q, a = sample
    keys = list(map(str, range(1, 10)))
    values = random.sample(keys, len(keys))
    mapping = dict(zip(keys, values))
    mapping['0'] = '0'

    def perm(digits):
        return ''.join([mapping[x] for x in digits])

    return perm(q), perm(a)


def add(sample, n):
    q, a = sample
    for _ in range(n):
        empty_places = [k for k, v in enumerate(q) if v == '0']
        idx = random.choice(empty_places)
        q = q[:idx] + a[idx] + q[(idx + 1):]
    return q, a


def generate(pool, n_extra_givens_max, n_per_givens):
    generated = []
    for k in range(n_extra_givens_max + 1):
        for i in range(n_per_givens):
            sample = random.choice(pool)
            sample = permute(sample)
            sample = add(sample, k)
            generated.append(sample)

    random.shuffle(generated)
    return generated


n_extra_givens_max = 17
train = generate(train_pool, n_extra_givens_max, 10000)
valid = generate(valid_pool, n_extra_givens_max, 1000)
test = generate(test_pool, n_extra_givens_max, 1000)


def dump(fname, samples):
    with open(fname, 'w') as f:
        for q, a in samples:
            f.write(q + "," + a + "\n")


dump('train.csv', train)
dump('valid.csv', valid)
dump('test.csv', test)
