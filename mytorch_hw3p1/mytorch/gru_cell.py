import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h
        #print(np.dot(self.Wrx,np.reshape(self.x, (self.x.shape[0],1))))
        self.r=self.r_act(np.dot(self.Wrx,self.x)+self.bir+np.dot(self.Wrh,self.hidden)+self.bhr)
        self.z=self.z_act(np.dot(self.Wzx,self.x)+self.biz+np.dot(self.Wzh,self.hidden)+self.bhz)
        self.n=self.h_act(np.dot(self.Wnx,self.x)+self.bin+(self.r*(np.dot(self.Wnh,self.hidden)+self.bhn)))
        h_t=((1-self.z)*self.n)+(self.z*self.hidden)
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
        raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        dh_t=delta
        dz=dh_t*(self.hidden-self.n)
        dn=dh_t*(1-self.z)
        dh=dh_t*self.z
        self.dWnx=np.dot(np.reshape(self.x,(self.d,1)),dn*(1-(self.n**2))).T
        self.dbin=dn*(1-(self.n**2))
        dx=np.dot(dn*(1-(self.n**2)),self.Wnx)
        dr=dn*(1-(self.n**2))*(np.dot(self.Wnh,self.hidden)+self.bhn)
        self.dWnh=np.dot(np.reshape(self.hidden,(self.h,1)),dn*(1-(self.n**2))*self.r).T
        dh+=np.dot(dn*(1-(self.n**2))*self.r,self.Wnh)
        self.dbhn=dn*(1-(self.n**2))*self.r
        self.dWzx=np.dot(np.reshape(self.x, (self.d,1)),dz*self.z*(1-self.z)).T
        dx+=np.dot(dz*self.z*(1-self.z), self.Wzx)
        self.dbiz=self.dbhz=dz*self.z*(1-self.z)
        self.dWzh=np.dot(np.reshape(self.hidden, (self.h,1)),dz*self.z*(1-self.z)).T
        dh+=np.dot(dz*self.z*(1-self.z),self.Wzh)
        self.dWrx=np.dot(np.reshape(self.x, (self.d,1)),dr*self.r*(1-self.r)).T
        dx+=np.dot(dr*self.r*(1-self.r), self.Wrx)
        self.dbir=self.dbhr=dr*self.r*(1-self.r)
        self.dWrh=np.dot(np.reshape(self.hidden, (self.h,1)),dr*self.r*(1-self.r)).T
        dh+=np.dot(dr*self.r*(1-self.r),self.Wrh)
        
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        raise NotImplementedError
