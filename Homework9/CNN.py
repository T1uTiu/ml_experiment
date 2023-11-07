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
    c = -np.sum(Y*np.log(Y_hat+1e-9)) / n
    
    return c

def test(Y_hat, Y):
    Y_out = np.zeros_like(Y)
    
    idx = np.argmax(Y_hat, axis=1)
    Y_out[range(Y.shape[0]),idx] = 1
    acc = np.sum(Y_out*Y) / Y.shape[0]
    print("Training accuracy is: %f" %(acc))
      
    return acc


iteration = 500###### Training loops
lr = 0.00075###### Learning rate

data = np.load("data.npy")

X = data[:,:-1].reshape(data.shape[0],1, 20, 20).transpose(0,1,3,2)
Y = data[:,-1].astype(np.int32)
(batchsize, in_channel, L, _) = X.shape
Y = onehotEncoder(Y, 10)

class ConvolutionLayer:
    def __init__(self, in_shape, out_channels, kernel_size, padding,stride):
        self.kernel_size, self.padding, self.stride = kernel_size, padding, stride
        self.out_channels = out_channels
        self.batchsize, self.in_channels, self.l_in, _ = in_shape

        self.W = np.random.standard_normal(0(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.b = np.random.rand(self.out_channels)

        self.relu = lambda x: max(0,x)
    
    def img2col(self, X, kernel_size, padding, stride):
        batchsize, in_c, l_in, l_in = X.shape
        l_out = (l_in+2*padding-kernel_size)//stride + 1
        
        X = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)), 'constant')
        col = np.zeros((batchsize, in_c, kernel_size, kernel_size, l_out, l_out))

        for y in range(kernel_size):
            y_max = y + stride*l_out
            for x in range(kernel_size):
                x_max = x + stride*l_out
                col[:,:,y,x,:,:] = X[:,:,y:y_max:stride,x:x_max:stride]
        
        col = col.transpose(0,4,5,1,2,3).reshape(batchsize*l_out*l_out, -1)

        return col

    def col2img(self, col, X_shape, kernel_size, padding, stride):
        batchsize, in_c, l_in, l_in = X_shape
        l_out = (l_in+2*padding-kernel_size)//stride + 1

        col = col.reshape(batchsize, l_out, l_out, in_c, kernel_size, kernel_size).transpose(0,3,4,5,1,2)
        X = np.zeros((batchsize, in_c, l_in+2*padding+stride-1, l_in+2*padding+stride-1))

        for y in range(kernel_size):
            y_max = y + stride*l_out
            for x in range(kernel_size):
                x_max = x + stride*l_out
                X[:,:,y:y_max:stride,x:x_max:stride] += col[:,:,y,x,:,:]
        
        return X[:, :, padding:l_in+padding, padding:l_in+padding]

    def forward(self, X):
        self.X = X
        l_out = (self.l_in+2*self.padding-self.kernel_size)//self.stride + 1

        self.X_col = self.img2col(X, self.kernel_size, self.padding, self.stride) # shape: [n*l_out*l_out, in_c*k*k]
        self.W_col = self.W.reshape(self.out_channels, -1).T # shape: [in_c*k*k, out_c]
        self.Y = self.X_col@self.W_col + self.b
        self.Y = self.Y.reshape(self.batchsize, l_out, l_out,self.out_channels).transpose(0,3,1,2) # shape: [n, out_c, l_out, l_out]
        return self.Y
    
    
    def backward(self, dz, lr=0.01):
        # dz shape: [n, out_c, l_out, l_out]
        dz_col = dz.transpose(0,2,3,1).reshape(-1, self.out_channels) # shape: [n*l_out*l_out, out_c]
        dW = np.dot(self.X_col.T, dz_col).T.reshape(self.W.shape) # shape: [in_c*k*k, out_c] to [out_c,in_c*k*k] to [out_c, in_c, k, k]
        db = np.sum(dz_col, axis=0)
        self.W -= lr * dW / self.batchsize
        self.b -= lr * db / self.batchsize

        dX_col = np.dot(dz_col, self.W_col.T) # shape: [n*l_out*l_out, in_c*k*k]
        dX = self.col2img(dX_col, self.X.shape, self.kernel_size, self.padding, self.stride)
        
        return dX


        
class MaxPoolingLayer:
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
    
    def backward(self, dz):
        dz = dz.reshape(self.batchsize, self.in_channels, self.l_out, self.l_out)
        return np.repeat(np.repeat(dz, self.stride, axis=2), self.stride, axis=3) * self.index

class LinearLayer:
    def __init__(self, n_in, n_out):
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
        self.W -= lr * dW / n
        self.b -= lr * db / n
        
        return dX

class ReluLayer:
    def __init__(self, leaky = 0):
        self.leak = leaky
    
    def forward(self, X):
        # leakyRelu
        self.X = X
        return np.maximum(X, self.leak*X)
    
    def backward(self, dz):
        # leakyRelu
        dz[self.X < 0] *= self.leak
        return dz


def softmax(x):
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T

conv1 = ConvolutionLayer(X.shape, out_channels=6, kernel_size=5, padding=0, stride=1)
relu1 = ReluLayer(0.01)
pool1 = MaxPoolingLayer((batchsize, 6, 16, 16), kernel_size=2)

conv2 = ConvolutionLayer((batchsize, 6, 8, 8), out_channels=16, kernel_size=5, padding=0, stride=1)
relu2 = ReluLayer(0.01)
pool2 = MaxPoolingLayer((batchsize, 16, 4, 4), kernel_size=2)

linear1 = LinearLayer(16*2*2, 120)
relu3 = ReluLayer(0.01)
linear2 = LinearLayer(120,84)
relu4 = ReluLayer(0.01)
linear3 = LinearLayer(84,10)
costs = []
def train():
    for i in range(iteration):
        z = pool1.forward(relu1.forward(conv1.forward(X)))
        z = pool2.forward(relu2.forward(conv2.forward(z)))
        z = relu3.forward(linear1.forward(z))
        z = relu4.forward(linear2.forward(z))
        z = linear3.forward(z)
        y_hat = softmax(z)

        dz = y_hat - Y
        dz = linear3.backward(dz, lr)
        dz = relu4.backward(dz)
        dz = linear2.backward(dz, lr)
        dz = relu3.backward(dz)
        dz = linear1.backward(dz, lr)
        dz = pool2.backward(dz)
        dz = relu2.backward(dz)
        dz = conv2.backward(dz, lr)
        dz = pool1.backward(dz)
        dz = relu1.backward(dz)
        dz = conv1.backward(dz, lr)

        costs.append(cost(y_hat, Y))
        print(f"iteration {i+1}: cost = {costs[-1]}")
            

train()
plt.plot(costs)
plt.show()

z = pool1.forward(relu1.forward(conv1.forward(X)))
z = pool2.forward(relu2.forward(conv2.forward(z)))
z = relu3.forward(linear1.forward(z))
z = relu4.forward(linear2.forward(z))
z = linear3.forward(z)
y_hat = softmax(z)

test(y_hat, Y)