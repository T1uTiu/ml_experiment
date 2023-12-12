
from optimizer.adam import Adam


class Net:
    def __init__(self):
        self.optimizer = Adam()
    def forward(self, X):
        pass
    def backward(self, dz, lr=0.01):
        pass