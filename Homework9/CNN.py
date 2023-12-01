# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from modules.conv2d import ConvolutionLayer
from modules.linear import LinearLayer
from modules.net import Net
from modules.pool2d import MaxPoolingLayer
from modules.relu import ReluLayer
from modules.residual import ResidualBlock

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




data = np.load("data.npy")

X = data[:,:-1].reshape(data.shape[0],1, 20, 20).transpose(0,1,3,2)
Y = data[:,-1].astype(np.int32)
Y = onehotEncoder(Y, 10)


def softmax(x):
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T


(n, in_channel, L, _) = X.shape

epoch = 10000###### Training loops
lr = 0.00001###### Learning rate
batchsize = 100###### Batch size
iteration = n // batchsize###### Number of batches in one epoch

costs = []

cnn_seq:list[Net] = [
    ConvolutionLayer((batchsize, 1, 20, 20), out_channels=6, kernel_size=5, padding=0, stride=1),
    ReluLayer(0.01),
    MaxPoolingLayer((batchsize, 6, 16, 16), kernel_size=2),

    ConvolutionLayer((batchsize, 6, 8, 8), out_channels=16, kernel_size=3, padding=0, stride=1),
    ReluLayer(0.01),
    MaxPoolingLayer((batchsize, 16, 6, 6), kernel_size=2),

    ConvolutionLayer((batchsize, 16, 3, 3), out_channels=24, kernel_size=3, padding=1, stride=1),
    ReluLayer(0.01),

    LinearLayer(24*3*3, 120),
    ReluLayer(0.01),
    LinearLayer(120,84),
    ReluLayer(0.01),
    LinearLayer(84,10)
]

def train():
    for i in range(epoch):
        c = 0
        for j in range(iteration):
            X_batch = X[j*batchsize:(j+1)*batchsize]
            Y_batch = Y[j*batchsize:(j+1)*batchsize]
            z = cnn_seq[0].forward(X_batch)
            for layer in cnn_seq[1:]:
                z = layer.forward(z)
            y_hat = softmax(z)
            dz = y_hat - Y_batch
            for layer in cnn_seq[::-1]:
                dz = layer.backward(dz, lr)

            c += cost(y_hat, Y_batch)

        costs.append(c/iteration)
        print(f"epoch {i+1}: cost = {costs[-1]}")

train()
plt.plot(costs)
plt.show()

# test(y_hat, Y)