import numpy as np
from .net import Net

class ReluLayer(Net):
    def __init__(self, leaky = 0):
        self.leak = leaky
    
    def forward(self, X):
        # leakyRelu
        self.X = X
        return np.maximum(X, self.leak*X)
    
    def backward(self, dz, lr=0.01):
        # leakyRelu
        dz = dz.reshape(self.X.shape)
        dz[self.X < 0] *= self.leak
        return dz