# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    tmp = np.exp(z)
    return tmp/(np.sum(tmp,axis=1)[:,np.newaxis]*np.ones((1,3)))

def cost_gradient(W, X, Y, n):
    z = X@W
    y_hat = softmax(z)
    G = X.T@(y_hat-Y) /n ###### Gradient
    j = (-Y*np.log(y_hat)).mean() ###### cost with respect to current W

    return (j, G)

def train(W, X, Y, n, lr, iterations):
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)
        W = W - lr*G

    return (W,J)

def error(W, X, Y):
    z = X@W
    Y_hat = softmax(z)###### Output Y_hat by the trained model
    pred = np.argmax(Y_hat, axis=1)
    label = np.argmax(Y, axis=1)
    
    return (1-np.mean(np.equal(pred, label)))

iterations = 5000###### Training loops
lr = 0.1###### Learning rate

data = np.loadtxt('SR.txt', delimiter=',')

n = data.shape[0]
X = np.concatenate([np.ones([n, 1]),
                    np.expand_dims(data[:,0], axis=1),
                    np.expand_dims(data[:,1], axis=1),
                    np.expand_dims(data[:,2], axis=1)],
                   axis=1)
Y = data[:, 3].astype(np.int32)
c = np.max(Y)+1
Y = np.eye(c)[Y]

W = np.random.random([X.shape[1], c])

(W,J) = train(W, X, Y, n, lr, iterations)

plt.figure()
plt.plot(range(iterations), J)
plt.show()

print(error(W,X,Y))