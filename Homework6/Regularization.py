# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]
    
    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]),
                        np.expand_dims(np.power(data[:,0],1), axis=1),
                        np.expand_dims(np.power(data[:,1],1), axis=1),
                        np.expand_dims(np.power(data[:,2],1), axis=1),
                        np.expand_dims(np.power(data[:,3],1), axis=1),
                        np.expand_dims(np.power(data[:,4],1), axis=1),
                        np.expand_dims(np.power(data[:,5],1), axis=1),
                        np.expand_dims(np.power(data[:,6],1), axis=1),
                        np.expand_dims(np.power(data[:,7],1), axis=1)],
                        axis=1)
    for i in range(8):
        X = np.concatenate([X, np.expand_dims(data[:,i]**2, axis=1)], axis=1)
    for i in range(8):
        X = np.concatenate([X, np.expand_dims(data[:,i]**3, axis=1)], axis=1)
    for i in range(8):
        X = np.concatenate([X, np.expand_dims(data[:,i]**4, axis=1)], axis=1)

    
    ###### You may modify this section to change the model

    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, -1], axis=1)
    
    return (X,Y,n)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cost_gradient(W, X, Y, n, lambd):
    Y_hat = sigmoid(X@W)
    G = X.T@(Y_hat-Y)/n + lambd*W ###### Gradient
    j = 0.5*np.sum(-Y*np.log(Y_hat)-(1-Y)*np.log(1-Y_hat))/n + 0.5*lambd*np.sum(np.power(W,2))###### cost with respect to current W
    
    return (j, G)

def train(W, X, Y, lr, n, iterations, lambd):
    J = np.zeros([iterations, 1])
    
    for i in range(iterations):
        lr *= 0.99
        (J[i], G) = cost_gradient(W, X, Y, n, lambd)
        W = W - lr*G
    err = error(W, X, Y)
    
    return (W,J,err)

def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    return (1-np.mean(np.equal(Y_hat, Y)))

def predict(W):
    (X, _, _) = read_data("test_data.csv")
    
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    idx = np.expand_dims(np.arange(1,201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header = "Index,ID", comments='', delimiter=',')
    
iterations = 2000 ###### Training loops
lr = 0.25 ###### Learning rate
lambd = 0.0005 ##### Lambda to control the weight of Regularization part

(X, Y, n) = read_data("train.csv")
W = np.random.random([X.shape[1], 1])

(W,J,err) = train(W, X, Y, lr, n, iterations, lambd)
print(err)

plt.figure()
plt.plot(range(iterations), J)
plt.show()

predict(W)