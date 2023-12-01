# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def update(W, X, Y, n, lr, iterations):
    J = np.zeros([iterations, 1])
      
    for i in range(iterations):
        Z = X@W
        # W -= lr*(-X.T@Y) / n ###### Update W in each iteration
        # J[i] = -np.sum(np.maximum(0, Z)* Y)###### Store the cost in each iteration
        for i in range(n):
            if Y[i, :][0]*Z[i,:][0] < 0:
                W -= lr*(-X[i, :]*Y[i, :])[:, np.newaxis]
                J[i] += -Y[i, :][0]*Z[i, :][0]

    return (W,J)

data = np.loadtxt('Perceptron.txt', delimiter=',')

n = data.shape[0]
W = np.random.random([3, 1])
X = np.concatenate([np.ones([n, 1]), data[:,0:2]], axis=1)
Y = np.expand_dims(data[:, 2], axis=1)

iterations = 100 #### Training loops
lr = 0.01 ###### Learning rate

(W,J) = update(W, X, Y, n, lr, iterations)

#Draw figure
idx0 = (data[:, 2]==-1)
idx1 = (data[:, 2]==1)

plt.figure()
plt.ylim(-12,12)
plt.plot(data[idx0,0], data[idx0,1],'go')
plt.plot(data[idx1,0], data[idx1,1],'rx')

x1 = np.arange(-10,10,0.2)
y1 = W[0] + W[1]*x1 / -W[2]
plt.plot(x1, y1)

plt.figure()
plt.plot(range(iterations), J)
plt.show()