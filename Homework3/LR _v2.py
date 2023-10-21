# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
    z = X.dot(W)
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

iterations = 250
lr = 0.006

data = np.loadtxt('LR.txt', delimiter=',')

n = data.shape[0]
W = np.random.random([4, 1])
# X = np.concatenate([data[:,0:1],data[:,0:1],np.ones([n, 1]),data[:,1:2]],axis=1)
X = np.ones((n,4))
X[:,1] = data[:,0]
X[:,2] = data[:,1]
X[:,3] = data[:,0]**2+ data[:,1]**2
# X = np.concatenate([np.ones([n, 1]), data[:,0:2], data[:,0]**2+data[:,1]**2], axis=1)
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
x2 = np.arange(-10,10,0.2)
x1, x2 = np.meshgrid(x1, x2)
z = W[0]+W[1]*x1+W[2]*x2+W[3]*(x1**2+x2**2)
plt.contour(x1, x2, z, levels=[0], colors='r', linestyles='dashed')
# y1 = (W[0] + W[1]*x1) / -W[2]
# y2 = (W[0]*x1**2+W[2]+W[1]*x1) / -W[3]
# plt.plot(x1, y1)
# plt.plot(x1,y2)

plt.figure()
plt.plot(range(iterations), J)

plt.show()