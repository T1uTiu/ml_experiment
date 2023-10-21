# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#Utilities
def onehotEncoder(Y, ny):
      return np.eye(ny)[Y]

#Xavier Initialization
def initWeights(M):
      l = len(M)
      W = []
      B = []
      
      for i in range(1, l):
            W.append(np.random.randn(M[i-1], M[i]))
            B.append(np.zeros([1, M[i]]))
            
      return W, B

def sigmoid(x):
      return 1 / (1+np.exp(-x))

def softmax(z):
    tmp = np.exp(z)
    return tmp/(np.sum(tmp,axis=1)[:,np.newaxis]*np.ones((1,tmp.shape[1])))

#Forward propagation
def networkForward(X, W, B):
      l = len(W)

      A = [None for _ in range(l+1)]
      A[0] = X
      
      ##### Calculate the output of every layer A[i], where i = 0, 1, 2, ..., l
      for i in range(1,l):
            A[i] = sigmoid(A[i-1]@W[i-1] + B[i-1])
      A[-1] = softmax(A[-2]@W[-1] + B[-1])

      return A
#--------------------------

#Backward propagation
def networkBackward(Y, A, W):
      l = len(W)
      dW = [None for _ in range(l)]
      dB = [None for _ in range(l)]
      
      ##### Calculate the partial derivatives of all w and b in each layer dW[i] and dB[i], where i = 1, 2, ..., l
      g = A[-1]-Y
      for i in range(l-1,-1,-1):
            dW[i] = A[i].T@g
            dB[i] = np.sum(g, axis=0)
            g = A[i]*(1-A[i])*(g@W[i].T)

      return dW, dB
#--------------------------

#Update weights by gradient descent
def updateWeights(W, B, dW, dB, lr):
      l = len(W)

      for i in range(l):
            W[i] = W[i] - lr*dW[i]
            B[i] = B[i] - lr*dB[i]

      return W, B

#Compute regularized cost function
def cost(A_l, Y, W):
      n = Y.shape[0]
      c = -np.sum(Y*np.log(A_l)) / n

      return c

def train(X, Y, M, lr = 0.1, iterations = 3000):
      costs = []
      W, B = initWeights(M)

      for i in range(iterations):
            A = networkForward(X, W, B)
            c = cost(A[-1], Y, W)
            dW, dB = networkBackward(Y, A, W)
            W, B = updateWeights(W, B, dW, dB, lr)

            if i % 10 == 0:
                  print("Cost after iteration %i: %f" %(i, c))
                  costs.append(c)

      return W, B, costs

def predict(X, W, B, Y):
      Y_out = np.zeros([X.shape[0], Y.shape[1]])
      
      A = networkForward(X, W, B)
      idx = np.argmax(A[-1], axis=1)
      Y_out[range(Y.shape[0]),idx] = 1
      
      return Y_out

def test(Y, X, W, B):
      Y_out = predict(X, W, B, Y)
      acc = np.sum(Y_out*Y) / Y.shape[0]
      print("Training accuracy is: %f" %(acc))
      
      return acc

iterations = 1000 ###### Training loops
lr = 0.00075 ###### Learning rate

data = np.load("data.npy")

X = data[:,:-1]
Y = data[:,-1].astype(np.int32)
(n, m) = X.shape
Y = onehotEncoder(Y, 10)

M = [400, 25,10,25, 10]
W, B, costs = train(X, Y, M, lr, iterations)

plt.figure()
plt.plot(range(len(costs)), costs)
plt.show()

test(Y, X, W, B)
