class Model:
    def __init__(self):
        super(Model, self).__init__()

    def train_batch(self):
        raise NotImplementedError()

    def val_batch(self):
        raise NotImplementedError()

    def load(self, name):
        raise NotImplementedError()

    def save(self, name):
        raise NotImplementedError()
