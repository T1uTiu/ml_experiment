import numpy

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update(self, W, dW, lr):
        if not hasattr(self, 'm'):
            self.v = numpy.zeros_like(W)
            self.s = numpy.zeros_like(W)

        self.t += 1
        self.v = self.beta1*self.v + (1-self.beta1)*dW
        self.s = self.beta2*self.s + (1-self.beta2)*(dW**2)
        v_hat = self.v / (1-self.beta1**self.t)
        s_hat = self.s / (1-self.beta2**self.t)
        W = W - lr*v_hat/(numpy.sqrt(s_hat)+self.epsilon)

        return W