# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
import numpy as np

def entropy(Y):
    keys, counts = np.unique(Y, return_counts=True)
    n = len(Y)
    res = 0
    for i in range(len(keys)):
        res += -counts[i]/n*np.log2(counts[i]/n)
    return res

def conditional_entropy(Y, X):
    n, d = X.shape
    res = [0]*d
    for i in range(d):
        keys, counts = np.unique(X[:,i], return_counts=True)
        for j in range(len(keys)):
            counts_y = Counter()
            for k in range(n):
                if X[k,i] == keys[j]:
                    counts_y[Y[k]] += 1
            entropy = sum(cnt/counts[j]*np.log2(cnt/counts[j]) for cnt in counts_y.values()) * -counts[j]/n
            res[i] += entropy        
    return res

def select_feature(Y, X):
    d = X.shape[1]
    CEs = [None for i in range(d)]
    
    entropy_Y = entropy(Y)
    conditional_entropy_Y_X = conditional_entropy(Y, X)
    for i in range(d):
        CEs[i] = entropy_Y - conditional_entropy_Y_X[i]###### Information gain or information gain ratio
        
    return np.argmax(CEs)

def delete_feature(X, features, x_i):
    X[:,x_i] = X[:,-1]
    features[x_i] = features[-1]
    
    return X[:,:-1], features[:-1]

def ID3(Y, X, features):
    keys, counts = np.unique(Y, return_counts=True)
    if keys.shape[0] == 1:
        return keys[0]
    elif X.shape[1] == 1:
        return keys[np.argmax(counts)]
    
    x_i = select_feature(Y, X)
    x_feature = features[x_i]
    tree = {x_feature: {}}
    
    idx = []
    keys = np.unique(X[:,x_i])
    for k in keys:
        idx.append(X[:,x_i]==k)
        
    X, features = delete_feature(X, features, x_i)
    for k, i in zip(keys, idx):
        tree[x_feature][k] = ID3(Y[i], X[i], features)
        
    return tree

def predict(X, features, tree):
    feature = list(tree.keys())[0]
    tree = tree[feature]
    
    idx = np.where(features==feature)[0]
    value = X[idx].item()
    tree = tree[value]
    
    if isinstance(tree, dict):
        return predict(X, features, tree)
    else:
        return tree

def test(Y, X, features, tree):
    Y_hat = np.zeros(X.shape[0], dtype="str")
    
    for i in range(X.shape[0]):
        Y_hat[i] = predict(X[i], features, tree)
    
    return np.mean(Y_hat == Y)

data = np.loadtxt('test.txt', dtype="str", delimiter=',')
features = data[0,:-1]
X = data[1:,:-1]
Y = data[1:,-1]

tree = ID3(Y, X, features)
print(tree)

data = np.loadtxt('test.txt', dtype="str", delimiter=',')
features = data[0,:-1]
X = data[1:,:-1]
Y = data[1:,-1]

print(test(Y, X, features, tree))

