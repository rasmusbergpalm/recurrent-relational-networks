import csv
import os
import urllib.request
import zipfile


class sudoku:
    url = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"  # See generate_hard.py on how this dataset was generated
    zip_fname = "/tmp/sudoku-hard.zip"
    dest_dir = '/tmp/sudoku-hard/'

    def __init__(self):
        if not os.path.exists(self.dest_dir):
            print("Downloading data...")

            urllib.request.urlretrieve(self.url, self.zip_fname)
            with zipfile.ZipFile(self.zip_fname) as f:
                f.extractall('/tmp/')

        def read_csv(fname):
            print("Reading %s..." % fname)
            with open(self.dest_dir + fname) as f:
                reader = csv.reader(f, delimiter=',')
                return [(q, a) for q, a in reader]

        self.train = read_csv('train.csv')
        self.valid = read_csv('valid.csv')
        self.test = read_csv('test.csv')
