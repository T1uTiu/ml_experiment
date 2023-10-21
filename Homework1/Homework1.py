import numpy as np

def problem1(n):
    matrix = np.random.random((1,n))
    print(matrix)
    print(matrix[:,::-1])

def problem2():
    matrix = np.random.random((10,10))
    print(matrix)
    print("ans:")
    print(np.max(matrix, axis=0))

def problem3():
    matrix = np.random.random((10,10))
    res = np.where(matrix > 0.5, 1, 0)
    print(matrix)
    print(res)

def problem4():
    matrix = np.random.random((10,10))
    print(matrix)
    print("mean:")
    print(np.mean(matrix, axis=0))
    print("std:")
    print(np.std(matrix, axis=0))

def problem5():
    tensor = np.random.random((3,5,5))
    matrix = np.random.random((5,5))
    res = tensor*matrix[np.newaxis,:,:]
    print("5*5*3:")
    print(tensor)
    print("5*5:")
    print(matrix)
    print("ans:")
    print(res)


problem5()
