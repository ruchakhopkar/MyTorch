# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, train=True):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          train (boolean): whether the model is in training mode
        Return:
          np.array of same shape as input x
        """
        if train:
             self.mask=np.random.binomial(np.ones((x.shape[0], x.shape[1]), dtype='int32'), 1-self.p)
             for i in range((x.shape[0])):
                 for j in range(x.shape[1]):
                     x[i,j,:,:]=np.multiply(x[i,j,:,:],self.mask[i,j])*(1/(1-self.p))
        return x
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.

        raise NotImplementedError

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        
        # 1) This method is only called during training.
        for i in range((delta.shape[0])):
                 for j in range(delta.shape[1]):
                     for k in range(delta.shape[2]):
                         for m in range(delta.shape[3]):
                             delta[i,j,k,m]*=self.mask[i,j]*(1/(1-self.p))
        return delta
        raise NotImplementedError

