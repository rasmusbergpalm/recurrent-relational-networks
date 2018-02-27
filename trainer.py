import time

from model import Model


def train(model: Model):
    n_updates = 5000000
    val_interval = 1000

    start = time.time()
    best = float("inf")
    for i in range(n_updates):
        loss = model.train_batch()

        if i % val_interval == 0:
            took = time.time() - start
            print("%05d/%05d %f updates/s %f loss" % (i, n_updates, val_interval / took, loss))
            val_loss = model.val_batch()
            if val_loss < best:
                best = val_loss
                model.save('./best')
            start = time.time()
