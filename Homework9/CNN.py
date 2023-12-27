# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from modules.conv2d import ConvolutionLayer
from modules.linear import LinearLayer
from modules.net import Net
from modules.pool2d import MaxPoolingLayer
from modules.relu import ReluLayer

plt.ion()

#Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]


#Compute the cost function
def cost(Y_hat, Y):
    n = Y.shape[0]
    c = -np.sum(Y*np.log(Y_hat+1e-9)) / n
    
    return c

def test(model: list[Net], X,  Y):
    z = model[0].forward(X)
    for layer in model[1:]:
        z = layer.forward(z)
    Y_hat = softmax(z)

    Y_out = np.zeros_like(Y)
    
    idx = np.argmax(Y_hat, axis=1)
    Y_out[range(Y.shape[0]),idx] = 1
    acc = np.sum(Y_out*Y) / Y.shape[0]
    print("Training accuracy is: %f" %(acc))
      
    return acc

def predict(model: list[Net], X):
    z = model[0].forward(X)
    for layer in model[1:]:
        z = layer.forward(z)
    Y_hat = softmax(z)
    Y_hat = np.argmax(Y_hat, axis=1,keepdims=True)
    n = X.shape[0]
    idx = np.expand_dims(np.arange(0,n), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header = "Index,ID", comments='', delimiter=',')


# data = np.load("data.npy")

# X = data[:,:-1].reshape(data.shape[0],1, 20, 20).transpose(0,1,3,2)
# Y = data[:,-1].astype(np.int32)
# Y = onehotEncoder(Y, 10)
train_data = np.load("train_data.npy")
X = train_data.reshape(train_data.shape[0], 28, 28)[:, np.newaxis, :, :]
Y = np.load("train_label.npy")
Y = onehotEncoder(Y, 10)
test_data = np.load("test_data.npy")
test_X = test_data.reshape(test_data.shape[0], 28, 28)[:, np.newaxis, :, :]


def softmax(x):
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T


(n, in_channel, L, _) = X.shape

epoch = 10000###### Training loops
lr = 0.01###### Learning rate
batchsize = 10###### Batch size
iteration = n // batchsize###### Number of batches in one epoch
save_epoch = 20 # 保存模型的步长

costs = []

cnn_seq:list[Net] = [
    ConvolutionLayer((1, 28, 28), out_channels=32, kernel_size=3, padding=1, stride=1),
    ReluLayer(0),

    ConvolutionLayer((32, 28, 28), out_channels=32, kernel_size=3, padding=1, stride=1),
    ReluLayer(0),
    
    ConvolutionLayer((32, 28, 28), out_channels=64, kernel_size=3, padding=1, stride=1),
    ReluLayer(0),
    MaxPoolingLayer((64, 28, 28), kernel_size=2),

    ConvolutionLayer((64, 14, 14), out_channels=128, kernel_size=3, padding=1, stride=1),
    LinearLayer(128 * 14 * 14, 512),
    LinearLayer(512, 128),
    LinearLayer(128, 10)
]

def loadWeights():
    global cnn_seq, costs
    try:
        weights = np.load("weights/mymodel_numpy.npz")
    except:
        print("No saved model found!")
    else:
        for i, layer in enumerate(cnn_seq):
            if hasattr(layer, "W"):
                layer.W = weights[f"layer_W_{i}"]
                layer.b = weights[f"layer_b_{i}"]
        costs = np.load("weights/costs_12200.npy").tolist()
        print("Model loaded!")

def saveWeights(step):
    global cnn_seq, costs
    np.save(f"weights/costs_{step}.npy", costs)
    weights = dict()
    for j, layer in enumerate(cnn_seq):
        if hasattr(layer, "W"):
            weights[f"layer_W_{j}"] = layer.W
            weights[f"layer_b_{j}"] = layer.b
    np.savez(f"weights/weights_{step}.npz", **weights)
    print("Model saved!")

def train():
    global lr, cnn_seq, costs
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

        if i % save_epoch == 0 and i != 0:
            saveWeights(len(costs)*iteration)


loadWeights()
# train()
# plt.plot(costs)
# plt.show()

# test(cnn_seq, X, Y)
predict(cnn_seq, test_X)
