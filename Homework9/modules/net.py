
from optimizer.adam import Adam


class Net:
    def __init__(self):
        self.optimizerW = Adam()
        self.optimizerb = Adam()

    def forward(self, X):
        pass
    def backward(self, dz, lr=0.01):
        pass