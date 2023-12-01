import numpy as np
from .net import Net


class MaxPoolingLayer(Net):
    def __init__(self, in_shape, kernel_size, padding = 0):
        self.kernel_size, self.padding, self.stride = kernel_size, padding, kernel_size
        self.batchsize, self.in_channels, self.l_in, _ = in_shape
        self.l_out = (self.l_in+2*self.padding-self.kernel_size)//self.stride + 1
        

    def forward(self, X):
        self.X = X
        self.index = np.zeros(X.shape)
        self.Y = np.zeros((self.batchsize, self.in_channels, self.l_out, self.l_out))
        for i in range(self.batchsize):
            for c in range(self.in_channels):
                for j in range(self.l_out):
                    for k in range(self.l_out):
                        max_j, max_k = np.unravel_index(np.argmax(X[i,c, j*self.stride:j*self.stride+self.kernel_size, k*self.stride:k*self.stride+self.kernel_size]), (self.kernel_size, self.kernel_size))
                        self.Y[i,c,j,k] = X[i,c,j*self.stride+max_j, k*self.stride+max_k]
                        self.index[i,c,j*self.stride+max_j,k*self.stride+max_k] = 1
        return self.Y
    
    def backward(self, dz, lr=0.01):
        try:
            dz = dz.reshape(self.batchsize, self.in_channels, self.l_out, self.l_out)
            return np.repeat(np.repeat(dz, self.stride, axis=2), self.stride, axis=3) * self.index
        except:
            print("断点")