# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
	def __init__(self, p=0.5):
		# Dropout probability
		self.p = p

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x, train = True):
         # 1) Get and apply a mask generated from np.random.binomial
		 # 2) Scale your output accordingly
		 # 3) During test time, you should not apply any mask or scaling.
         if train:
             self.mask=np.random.binomial(np.ones(x.shape, dtype='int32'), 1-self.p)
             x=np.multiply(x,self.mask)*(1/(1-self.p))
         return x
         #raise NotImplementedError
		
		      


	def backward(self, delta):
         # 1) This method is only called during trianing.
         return np.multiply(self.mask, delta)
		
		
