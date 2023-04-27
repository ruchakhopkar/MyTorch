# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

"""
In the linear.py file, attributes have been added to the Linear class to make
implementing Adam easier, check them out!

self.mW = np.zeros(None) #mean derivative for W
self.vW = np.zeros(None) #squared derivative for W
self.mb = np.zeros(None) #mean derivative for b
self.vb = np.zeros(None) #squared derivative for b
"""

class adam():
    def __init__(self, model, beta1=0.9, beta2=0.999, eps = 1e-8):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = self.model.lr
        self.t = 0 # Number of Updates

    def step(self):
        '''
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        '''
        self.t += 1
        for i in range(len(self.model.linear_layers)):
            self.model.linear_layers[i].mW=(self.beta1*self.model.linear_layers[i].mW)+\
                ((1-self.beta1)*self.model.linear_layers[i].dW)
            self.model.linear_layers[i].vW=(self.beta2*self.model.linear_layers[i].vW)+\
                ((1-self.beta2)*np.square(self.model.linear_layers[i].dW))
            mWhat=self.model.linear_layers[i].mW/(1-(self.beta1**self.t))
            vWhat=self.model.linear_layers[i].vW/(1-(self.beta2**self.t))
            
            self.model.linear_layers[i].mb=(self.beta1*self.model.linear_layers[i].mb)+\
                ((1-self.beta1)*self.model.linear_layers[i].db)
            self.model.linear_layers[i].vb=(self.beta2*self.model.linear_layers[i].vb)+\
                ((1-self.beta2)*np.square(self.model.linear_layers[i].db))
            mbhat=self.model.linear_layers[i].mb/(1-(self.beta1**self.t))
            vbhat=self.model.linear_layers[i].vb/(1-(self.beta2**self.t))
            
            self.model.linear_layers[i].W-=self.lr*mWhat/(np.sqrt(vWhat+self.eps))
            self.model.linear_layers[i].b-=self.lr*mbhat/(np.sqrt(vbhat+self.eps))
        # Add your code here!

        #raise NotImplementedError
        
class adamW():
    def __init__(self, model, beta1=0.9, beta2=0.999, eps = 1e-8, weight_decay=0.01):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = self.model.lr
        self.t = 0 # Number of Updates
        self.weight_decay=weight_decay

    def step(self):
        '''
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        '''
        self.t += 1
        for i in range(len(self.model.linear_layers)):
            self.model.linear_layers[i].W-=self.model.linear_layers[i].W*self.lr*self.weight_decay
            self.model.linear_layers[i].b-=self.model.linear_layers[i].b*self.lr*self.weight_decay
            self.model.linear_layers[i].mW=(self.beta1*self.model.linear_layers[i].mW)+\
                ((1-self.beta1)*self.model.linear_layers[i].dW)
            self.model.linear_layers[i].vW=(self.beta2*self.model.linear_layers[i].vW)+\
                ((1-self.beta2)*np.square(self.model.linear_layers[i].dW))
            mWhat=self.model.linear_layers[i].mW/(1-(self.beta1**self.t))
            vWhat=self.model.linear_layers[i].vW/(1-(self.beta2**self.t))
            
            self.model.linear_layers[i].mb=(self.beta1*self.model.linear_layers[i].mb)+\
                ((1-self.beta1)*self.model.linear_layers[i].db)
            self.model.linear_layers[i].vb=(self.beta2*self.model.linear_layers[i].vb)+\
                ((1-self.beta2)*np.square(self.model.linear_layers[i].db))
            mbhat=self.model.linear_layers[i].mb/(1-(self.beta1**self.t))
            vbhat=self.model.linear_layers[i].vb/(1-(self.beta2**self.t))
            
            self.model.linear_layers[i].W-=self.lr*mWhat/(np.sqrt(vWhat+self.eps))
            self.model.linear_layers[i].b-=self.lr*mbhat/(np.sqrt(vbhat+self.eps))
        # Add your code here!

        #raise NotImplementedError
