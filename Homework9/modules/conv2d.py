import numpy as np
from .net import Net

class ConvolutionLayer(Net):
    def __init__(self, in_shape, out_channels, kernel_size, padding,stride):
        super().__init__()
        self.kernel_size, self.padding, self.stride = kernel_size, padding, stride
        self.out_channels = out_channels
        self.in_channels, self.l_in, _ = in_shape

        self.W = np.random.standard_normal((self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.b = np.random.rand(self.out_channels)
    
    def img2col(self, X, kernel_size, padding, stride):
        batchsize, in_c, l_in, l_in = X.shape
        l_out = (l_in+2*padding-kernel_size)//stride + 1
        
        X = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)), 'constant')
        col = np.zeros((batchsize, in_c, kernel_size, kernel_size, l_out, l_out))

        for y in range(kernel_size):
            y_max = y + stride*l_out
            for x in range(kernel_size):
                x_max = x + stride*l_out
                col[:,:,y,x,:,:] = X[:,:,y:y_max:stride,x:x_max:stride]
        
        col = col.transpose(0,4,5,1,2,3).reshape(batchsize*l_out*l_out, -1)

        return col

    def col2img(self, col, X_shape, kernel_size, padding, stride):
        batchsize, in_c, l_in, l_in = X_shape
        l_out = (l_in+2*padding-kernel_size)//stride + 1

        col = col.reshape(batchsize, l_out, l_out, in_c, kernel_size, kernel_size).transpose(0,3,4,5,1,2)
        X = np.zeros((batchsize, in_c, l_in+2*padding+stride-1, l_in+2*padding+stride-1))

        for y in range(kernel_size):
            y_max = y + stride*l_out
            for x in range(kernel_size):
                x_max = x + stride*l_out
                X[:,:,y:y_max:stride,x:x_max:stride] += col[:,:,y,x,:,:]
        
        return X[:, :, padding:l_in+padding, padding:l_in+padding]

    def forward(self, X):
        self.batchsize, _, _, _ = X.shape
        l_out = (self.l_in+2*self.padding-self.kernel_size)//self.stride + 1

        self.X_col = self.img2col(X, self.kernel_size, self.padding, self.stride) # shape: [n*l_out*l_out, in_c*k*k]
        self.W_col = self.W.reshape(self.out_channels, -1).T # shape: [in_c*k*k, out_c]
        Y = self.X_col@self.W_col + self.b
        Y = Y.reshape(self.batchsize, l_out, l_out,self.out_channels).transpose(0,3,1,2) # shape: [n, out_c, l_out, l_out]
        return Y
    
    
    def backward(self, dz, lr=0.01):
        # dz shape: [n, out_c, l_out, l_out]
        dz_col = dz.transpose(0,2,3,1).reshape(-1, self.out_channels) # shape: [n*l_out*l_out, out_c]
        dW = np.dot(self.X_col.T, dz_col).T.reshape(self.W.shape) # shape: [in_c*k*k, out_c] to [out_c,in_c*k*k] to [out_c, in_c, k, k]
        db = np.sum(dz_col, axis=0)
        self.W = self.optimizerW.update(self.W, dW/self.batchsize, lr)
        self.b = self.optimizerb.update(self.b, db/self.batchsize, lr)

        dX_col = np.dot(dz_col, self.W_col.T) # shape: [n*l_out*l_out, in_c*k*k]
        dX = self.col2img(dX_col, (self.batchsize, self.in_channels, self.kernel_size, self.kernel_size), self.kernel_size, self.padding, self.stride)
        
        return dX