# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
    z = np.dot(X,W)
    y_hat = 1/(1+np.exp(-z))
    G = X.T.dot(y_hat-Y)/n ###### Gradient
    j = np.sum(-Y*np.log(y_hat+1e-15)-(1-Y)*np.log(1-y_hat+1e-15)) ###### cost with respect to current W
      
    return (j, G)

def gradientDescent(W, X, Y, n, lr, iterations):
      J = np.zeros([iterations, 1])
      
      for i in range(iterations):
          (J[i], G) = cost_gradient(W, X, Y, n)
          W = W - lr*G###### Update W based on gradient

      return (W,J)

iterations = 200
lr = 0.2

data = np.loadtxt('LR.txt', delimiter=',')

n = data.shape[0]
W = np.random.random([3, 1])
X = np.concatenate([np.ones([n, 1]), data[:,0:2]], axis=1)
Y = np.expand_dims(data[:, 2], axis=1)

(W,J) = gradientDescent(W, X, Y, n, lr, iterations)

#Draw figure
idx0 = (data[:, 2]==0)
idx1 = (data[:, 2]==1)

plt.figure()
plt.ylim(-12,12)
plt.plot(data[idx0,0], data[idx0,1],'go')
plt.plot(data[idx1,0], data[idx1,1],'rx')

x1 = np.arange(-10,10,0.2)
y1 = (W[0] + W[1]*x1) / -W[2]
# y2 = -0.075*x1**2+0.35-0.2*x1
plt.plot(x1, y1)
# plt.plot(x1,y2)

plt.figure()
plt.plot(range(iterations), J)

plt.show()