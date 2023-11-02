# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

#Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]

#Compute the cost function
def cost(Y_hat, Y):
    n = Y.shape[0]
    c = -np.sum(Y*np.log(Y_hat)) / n
    
    return c

def test(Y_hat, Y):
    Y_out = np.zeros_like(Y)
    
    idx = np.argmax(Y_hat[-1], axis=1)
    Y_out[range(Y.shape[0]),idx] = 1
    acc = np.sum(Y_out*Y) / Y.shape[0]
    print("Training accuracy is: %f" %(acc))
      
    return acc


iteration = 100###### Training loops
lr = 0.1###### Learning rate
n_layer = 3###### The number of layers
n_filter = 1###### The number of convolutional kernels in each layer
kernel_size = 3###### The size of convolutional kernels
padding = 1###### The size of padding
stride = 1###### The size of stride
pool_size = 2###### The size of pooling kernels

data = np.load("data.npy")

X = data[:,:-1].reshape(data.shape[0],1, 20, 20).transpose(0,1,3,2)
Y = data[:,-1].astype(np.int32)
(batchsize, in_channel, L, _) = X.shape
Y = onehotEncoder(Y, 10)

class ConvolutionLayer:
    def __init__(self, in_shape, out_channels, kernel_size, padding,stride, lr=0.01):
        self.kernel_size, self.padding, self.stride = kernel_size, padding, stride
        self.out_channels = out_channels
        self.batchsize, self.in_channels, self.l_in, _ = in_shape
        self.lr = lr

        self.W = np.random.randn(self.out_channels, self.in_channels, kernel_size, kernel_size)
        self.b = np.random.rand(self.out_channels)

        self.relu = lambda x: max(0,x)
    
    def img2col(self, X, l_out, in_channels):
        # l_out = (self.l_in+2*self.padding-self.kernel_size)//self.stride + 1
        X_col = np.zeros((self.batchsize*l_out*l_out, in_channels*self.kernel_size*self.kernel_size))
        outsize = l_out*l_out
        for i in range(l_out):
            for j in range(l_out):
                X_col[i*l_out::outsize, :] = X[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size].reshape(self.batchsize, -1)
        return X_col

    def forward(self, X):
        # padding
        self.X = X
        X = np.zeros((self.batchsize, self.in_channels, self.l_in+2*self.padding, self.l_in+2*self.padding))
        X[:,:,self.padding:self.padding+self.l_in,self.padding:self.padding+self.l_in] = self.X

        self.l_out = (self.l_in+2*self.padding-self.kernel_size)//self.stride + 1

        self.X_col = self.img2col(X, self.l_out, self.in_channels)
        self.Y = np.dot(self.X_col, self.W.reshape(self.out_channels, -1).T) + self.b
        self.Y = np.maximum(self.Y, 0, self.Y).reshape(self.batchsize, self.out_channels, self.l_out, self.l_out)
        return self.Y
    
    
    def backward(self, dz):
        dz_col = dz.transpose(0,2,3,1).reshape(-1, self.out_channels)
        dW = np.dot(self.X_col.T, dz_col).T.reshape(self.W.shape)
        db = np.sum(dz_col, axis=0)
        self.W -= self.lr * dW / self.batchsize
        self.b -= self.lr * db / self.batchsize

        
        if self.l_out == self.l_in:
            padding = self.kernel_size // 2
        else:
            padding = self.kernel_size - 1
        dz_pad = np.zeros((self.batchsize, self.out_channels, self.l_out+2*padding, self.l_out+2*padding))
        dz_pad[:,:,padding:padding+self.l_out,padding:padding+self.l_out] = dz

        flip_w = np.flip(self.W, (2,3)).transpose(1,0,2,3)
        dz_col = self.img2col(dz_pad, self.l_in, self.out_channels)
        dX = np.dot(dz_col, flip_w.reshape(self.in_channels, -1).T)
        return dX.reshape(self.batchsize, self.in_channels, self.l_in, self.l_in)


        
class MaxPoolingLayer:
    def __init__(self, in_shape, kernel_size, stride, padding = 0):
        self.kernel_size, self.padding, self.stride = kernel_size, padding, kernel_size
        self.batchsize, self.in_channels, self.l_in, _ = in_shape
        self.l_out = (self.l_in+2*self.padding-self.kernel_size)//self.stride + 1
        self.index = np.zeros(in_shape)
        
    def forward(self, X):
        self.X = X
        self.Y = np.zeros((self.batchsize, self.in_channels, self.l_out, self.l_out))
        for i in range(self.batchsize):
            for c in range(self.in_channels):
                for j in range(self.l_out):
                    for k in range(self.l_out):
                        max_j, max_k = np.unravel_index(np.argmax(X[i,c, j*self.stride:j*self.stride+self.kernel_size, k*self.stride:k*self.stride+self.kernel_size]), (self.kernel_size, self.kernel_size))
                        self.Y[i,c,j,k] = X[i,c,j*self.stride+max_j, k*self.stride+max_k]
                        self.index[i,c,j*self.stride+max_j,k*self.stride+max_k] = 1
        return self.Y
    
    def backward(self, dz):
        dz = dz.reshape(self.batchsize, self.in_channels, self.l_out, self.l_out)
        return np.repeat(np.repeat(dz, self.stride, axis=2), self.stride, axis=3) * self.index

class LinearLayer:
    def __init__(self, n_in, n_out, lr=0.01):
        self.n_in = n_in
        self.n_out = n_out
        self.lr = lr
        self.W = np.random.randn(n_in, n_out)
        self.b = np.random.randn(n_out)
        
    def forward(self, X):
        # 把X, shape=(n, s, s)拉成一维向量
        self.X = X.reshape(X.shape[0], -1)
        self.Y = np.dot(self.X, self.W) + self.b
        return self.Y
    
    def backward(self, dz):
        (n, _) = self.X.shape
        dX = np.dot(dz, self.W.T)
        dW = np.dot(self.X.T, dz)
        db = np.sum(dz, axis=0)
        self.W -= self.lr * dW / n
        self.b -= self.lr * db / n
        return dX

def softmax(z):
    tmp = np.exp(z)
    return tmp/(np.sum(tmp,axis=1)[:,np.newaxis]*np.ones((1,tmp.shape[1])))

conv1 = ConvolutionLayer(X.shape, 3, kernel_size, padding, stride)
pool = MaxPoolingLayer((batchsize, 3, 20, 20), pool_size, stride)
linear = LinearLayer(100*3, 10)
def train():
    for i in range(iteration):
        z = conv1.forward(X)
        z = pool.forward(z)
        z= linear.forward(z)
        y_hat = softmax(z)

        dz = y_hat - Y
        dz = linear.backward(dz)
        dz = pool.backward(dz)
        dz = conv1.backward(dz)

        if i % 5 == 0:
            print(cost(y_hat, Y))

train()

# test(y_hat, Y)