import numpy as np
from .net import Net

class LinearLayer(Net):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.W = np.random.standard_normal((n_in, n_out))
        self.b = np.random.randn(n_out)
        
    def forward(self, X):
        # 把X, shape=(n, s, s)拉成一维向量
        self.X = X.reshape(X.shape[0], -1)
        self.Y = np.dot(self.X, self.W) + self.b
        return self.Y
    
    def backward(self, dz, lr=0.01):
        (n, _) = self.X.shape
        dX = np.dot(dz, self.W.T)
        dW = np.dot(self.X.T, dz)
        db = np.sum(dz, axis=0)
        self.W = self.optimizer.update(self.W, dW/n, lr)
        self.b = self.optimizer.update(self.b, db/n, lr)
        
        return dX