# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)
        self.X=None
        
        # TODO: Complete these but do not change the names.
        self.dW = np.zeros(None)
        self.db = np.zeros(None)
        
        self.momentum_W = np.zeros(None)
        self.momentum_b = np.zeros(None)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        affine=(np.dot(x, self.W)+self.b)
        self.X=x
        return affine
        #raise NotImplemented

    def backward(self, delta):

        """lp[l]
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
            
            
            
        """
        self.dW=(1/delta.shape[0])*np.dot((self.X).T, delta)
        self.db=(1/delta.shape[0])*np.dot(np.ones((1,delta.shape[0])), delta)
        dZ=np.dot(delta, self.W.T)
        return dZ
        #raise NotImplemented
