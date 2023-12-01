import time
import cupy as cp
import numpy as np

### Numpy and CPU
s = time.time()
x_cpu = np.ones((800,800,800))
e = time.time()
print(e - s)### CuPy and GPU
s = time.time()
x_gpu = cp.ones((10,10,10))
e = time.time()
print(e - s)
