import glob
import os
import tarfile
import urllib.request
from collections import namedtuple

import time
import random

from sgfmill.boards import Board
from sgfmill.sgf import Sgf_game
import numpy as np

Position = namedtuple('Position', ['states', 'winner', 'move'])


def sgf_files():
    raw_url = "https://www.dropbox.com/s/ck28dtozdh6loer/kgs-all-raw.tgz?dl=1"
    data_dir = os.environ.get('DATA_DIR') or '/tmp'
    tgz_fname = data_dir + '/kgs-all-raw.tgz'
    dir = data_dir + '/all'

    if not os.path.exists(dir) or not os.listdir(dir):
        print("Downloading data...")
        urllib.request.urlretrieve(raw_url, tgz_fname)

        print("Extracting...")
        with tarfile.open(tgz_fname) as tar:
            tar.extractall(path=data_dir)

    for sgf_fp in glob.glob(dir + '/*.sgf'):
        yield sgf_fp


def file_splits():
    random.seed(0)
    all = list(sgf_files())
    n_all = len(all)
    all = random.sample(all, n_all)
    n_test = int(n_all * 0.05)

    test = all[:n_test]
    val = all[n_test:2 * n_test]
    train = all[2 * n_test:]
    return train, val, test


def games(sgf_files):
    for sgf_fn in sgf_files:
        with open(sgf_fn) as fp:
            sgf = Sgf_game.from_string(fp.read())
            if sgf.get_size() == 19 and len(sgf.get_main_sequence()) > 10:
                yield sgf


def rnn_encoded(sgf_games):
    def encode_state(state, player):
        board = np.zeros((19, 19), dtype=np.float32)
        for color, (row, column) in state:
            board[row, column] = 1. if color == 'b' else -1.
        if player == 'w':
            board = -board

        return board.reshape((19 ** 2,))

    def encode_move(move):
        color, rc = move
        if rc is None:
            return 19 * 19  # pass
        else:
            row, column = rc
            return row * 19 + column

    for game in sgf_games:
        states = []
        moves = []
        values = []

        winner = game.get_winner()
        if winner is None:  # Very few games are actually draws. Most are in a weird state. We'll just skip them.
            continue

        board = Board(19)
        board.apply_setup(*game.root.get_setup_stones())
        seq = game.get_main_sequence()
        for move in seq[1:]:  # first move is weird root node
            move = move.get_move()
            moves.append(encode_move(move))
            player, rc = move
            states.append(encode_state(board.list_occupied_points(), player))
            if winner == player:
                values.append(1.)
            else:
                values.append(-1.)
            if rc is not None:  # not a pass
                (row, column) = rc
                board.play(row, column, player)

        yield np.array(states, dtype=np.float32), np.array(moves, dtype=np.int32), np.array(values, dtype=np.int32)


def positions(sgf_games):
    for game in sgf_games:
        states = []
        winner = game.get_winner()
        if winner is None:  # Very few games are actually draws. Most are in a weird state. We'll just skip them.
            continue

        board = Board(19)
        board.apply_setup(*game.root.get_setup_stones())
        moves = game.get_main_sequence()
        for move in moves[1:]:  # first move is weird root node
            move = move.get_move()
            states.append(board.list_occupied_points())
            yield Position(states, winner, move)
            color, rc = move
            if rc is not None:  # not a pass
                (row, column) = rc
                board.play(row, column, color)


def plane_encoded(positions):
    for pos in positions:
        n_history = 1
        black = np.zeros((19, 19, n_history), dtype=np.float32)
        white = np.zeros((19, 19, n_history), dtype=np.float32)
        for i, state in enumerate(reversed(pos.states[-n_history:])):
            for color, (row, column) in state:
                if color == 'b':
                    black[row, column, i] = 1.0
                elif color == 'w':
                    white[row, column, i] = 1.0
                else:
                    raise RuntimeError('Stone color should be black or white')

        if pos.winner == 'b':
            winner = 1.0
        elif pos.winner == 'w':
            winner = -1.0
        else:
            raise RuntimeError('Winner should be black or white')

        color, rc = pos.move
        if rc is None:
            action = 19 * 19  # pass
        else:
            row, column = rc
            action = row * 19 + column

        if color == 'b':
            planes = np.concatenate([black, white, np.ones((19, 19, 1), dtype=np.float32)], axis=2)
        elif color == 'w':
            planes = np.concatenate([white, black, np.zeros((19, 19, 1), dtype=np.float32)], axis=2)
        else:
            raise RuntimeError('Move color should be black or white')

        yield planes, winner, action


def graph_encoded(positions):
    for pos in positions:
        stones = np.zeros((19, 19), dtype=np.int32)
        for color, (row, column) in pos.states[-1]:
            stones[row, column] = 1 if color == 'b' else 2

        if pos.winner == 'b':
            winner = 1.0
        elif pos.winner == 'w':
            winner = -1.0
        else:
            raise RuntimeError('Winner should be black or white')

        color, rc = pos.move
        if rc is None:
            action = 19 * 19  # pass
        else:
            row, column = rc
            action = row * 19 + column

        color = ([0] if color == 'b' else [1]) * 19 ** 2

        yield stones.reshape((19 ** 2,)), color, winner, action


if __name__ == '__main__':
    start = time.time()
    for i, sample in enumerate(rnn_encoded(games(sgf_files()))):
        print([s.shape for s in sample])
        if i % 1000 == 0:
            now = time.time()
            print(i / (now - start), "samples/s")
