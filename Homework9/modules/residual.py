import numpy as np
from .conv2d import ConvolutionLayer
from .relu import ReluLayer
from .net import Net

class ResidualBlock(Net):
    def __init__(self, in_shape, out_channel, stride) -> None:
        assert stride in [1, 2]
        self.kernel_size, self.padding, self.stride = 3, 1, stride
        self.batchsize, self.in_channel, self.l_in, _ = in_shape
        l_out = (self.l_in+2*self.padding-self.kernel_size)//self.stride + 1
        self.out_shape = (self.batchsize, out_channel, l_out, l_out)

        self.conv1 = ConvolutionLayer(in_shape, out_channel, self.kernel_size, self.padding, self.stride)
        self.relu1 = ReluLayer(0.01)
        self.conv2 = ConvolutionLayer(self.out_shape, out_channel, self.kernel_size, self.padding, 1)
        self.conv3 = None
        self.relu2 = ReluLayer(0.01)
        if stride == 2 or self.in_channel != out_channel:
            self.conv3 = ConvolutionLayer(in_shape, out_channel, 1, 0, self.stride)

    def forward(self, X):
        self.X = X
        self.Y = self.conv1.forward(X)
        self.Y = self.relu1.forward(self.Y)
        self.Y = self.conv2.forward(self.Y)
        if self.conv3 is not None:
            self.Xc = self.conv3.forward(self.X)
            self.Y += self.Xc
        self.Y = self.relu2.forward(self.Y)
        return self.Y

    def backward(self, dZ, lr=0.01):
        dX = self.relu2.backward(dZ)
        dX3 = dX
        if self.conv3 is not None:
            dX3 = self.conv3.backward(dX, lr)
        dX = self.conv2.backward(dX, lr)
        dX = self.relu1.backward(dX)
        dX = self.conv1.backward(dX, lr)
        return dX + dX3
        