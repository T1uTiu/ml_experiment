# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, combinations_with_replacement
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split


def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]

    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]), data[:,0:6]/2], axis=1)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    for i in range(6):
        X = np.concatenate([X, np.expand_dims(data[:, i]**2, axis=1)], axis=1)
    for i in range(6):
        X = np.concatenate([X, np.expand_dims(data[:, i]**3, axis=1)], axis=1)
    for i in range(6):
        X = np.concatenate([X, np.expand_dims(data[:, i]**4, axis=1)], axis=1)
    for i in range(6):
        X = np.concatenate([X, np.expand_dims(np.exp(data[:, i]), axis=1)], axis=1)
    ###### You may modify this section to change the model
    
    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, 6], axis=1)
    
    return (X,Y,n)

def cost_gradient(W, X, Y, n):
    z = X.dot(W)
    y_hat = 1/(1+np.exp(-z))
    G = X.T.dot(y_hat-Y)/n ###### Gradient
    j = np.sum(-Y*np.log(y_hat+1e-15)-(1-Y)*np.log(1-y_hat+1e-15)) ###### cost with respect to current W
      
    return (j, G)

def train(W, X, Y, lr, n, iterations):
    ###### You may modify this section to do 10-fold validation
    J = np.zeros([iterations, 1])
    E_trn = np.zeros([iterations, 1])
    E_val = np.zeros([iterations, 1])
    n = int(0.9*n)
    X_trn, X_val,Y_trn,Y_val = train_test_split(X,Y,test_size=0.1, random_state=666)
    for i in range(iterations):
        lr *= 0.999
        (J[i], G) = cost_gradient(W, X_trn, Y_trn, n)
        W = W - lr*G
        E_trn[i] = error(W, X_trn, Y_trn)
        E_val[i] = error(W, X_val, Y_val)
    print(E_val[-1])

    # clf = LogisticRegressionCV(solver="saga",tol=0.1,penalty='l1')
    # clf.fit(X_trn, Y_trn)
    # print(clf.score(X_val,Y_val))
    # grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
    # grid.fit(X_trn, Y_trn)
    # print("The best parameters are %s with a score of %0.2f"
    #     % (grid.best_params_, grid.best_score_))

    ###### You may modify this section to do 10-fold validation

    return (W,J,E_trn,E_val)

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

iterations = 10000###### Training loops
lr = 0.2###### Learning rate

(X, Y, n) = read_data("train.csv")
W = np.random.random([X.shape[1], 1])


(W,J,E_trn,E_val) = train(W, X, Y, lr, n, iterations)
# print(W)

###### You may modify this section to do 10-fold validation
plt.figure()
plt.plot(range(iterations), J)
plt.figure()
plt.ylim(0,1)
plt.plot(range(iterations), E_trn, "b")
plt.plot(range(iterations), E_val, "r")
plt.show()
###### You may modify this section to do 10-fold validation

predict(W)
