"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()


# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------
        self.output_from_forward=None
        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        dims_layers=[input_size]+hiddens+[output_size]
        self.linear_layers = [Linear(in_feature, out_feature, weight_init_fn, bias_init_fn)\
                              for in_feature, out_feature in zip(dims_layers[:-1],\
                                                                 dims_layers[1:])]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(in_feature) for in_feature in dims_layers[1:num_bn_layers+1]]


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        for i in range(self.nlayers):
             x=self.linear_layers[i].forward(x) 
             
             if self.bn==True and i<self.num_bn_layers:
                 if self.train_mode==True:
                
                     x=self.bn_layers[i].forward(x)
                 else:
                     x=self.bn_layers[i].forward(x, eval=True)
             x=self.activations[i].forward(x)
                     
        self.output_from_forward=x
        return x
        #raise NotImplemented

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        #raise NotImplemented
        for i in range(self.nlayers):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)
            if self.bn==True and i<self.num_bn_layers:
                self.bn_layers[i].dgamma.fill(0.0)
                self.bn_layers[i].dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            if self.momentum:
                self.linear_layers[i].momentum_W=(self.momentum*self.linear_layers[i].momentum_W)\
                    -(self.lr*self.linear_layers[i].dW)
                self.linear_layers[i].W=self.linear_layers[i].W+self.linear_layers[i].momentum_W
                self.linear_layers[i].momentum_b=(self.momentum*self.linear_layers[i].momentum_b)\
                    -(self.lr*self.linear_layers[i].db)
                self.linear_layers[i].b=self.linear_layers[i].b+self.linear_layers[i].momentum_b
            else:
                self.linear_layers[i].W=self.linear_layers[i].W-(self.lr*self.linear_layers[i].dW)
                self.linear_layers[i].b=self.linear_layers[i].b-(self.lr*self.linear_layers[i].db)
            if self.bn==True and i<self.num_bn_layers:
                self.bn_layers[i].gamma=self.bn_layers[i].gamma-(self.lr*self.bn_layers[i].dgamma)
                self.bn_layers[i].beta=self.bn_layers[i].beta-(self.lr*self.bn_layers[i].dbeta)
            
        
        # Do the same for batchnorm layers

        #raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        
        
        loss=(self.criterion.forward(self.output_from_forward,labels))
        y=(self.criterion.derivative())
        
        for i in range(self.nlayers-1,-1,-1):
            z=(np.multiply(y, self.activations[i].derivative()))
            if self.bn==True and i<self.num_bn_layers:
                z=self.bn_layers[i].backward(z)
            y=(self.linear_layers[i].backward(z))
            
        #raise NotImplemented

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

#This function does not carry any points. You can try and complete this function to train your network.
def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):
        np.random.shuffle(idxs)
        xtrain=trainx[idxs]
        ytrain=trainy[idxs]
        
        # Per epoch setup ...
        mlp.train()
        for b in range(0, len(trainx), batch_size):

            # Remove this line when you start implementing this
            # Train ...
            
            mlp.zero_grads()
            ypred=mlp.forward(xtrain[b:b+batch_size])
            ytrue=ytrain[b:b+batch_size]
            mlp.backward(ytrue)
            mlp.step()
            sce=SoftmaxCrossEntropy()
            loss=sce.forward(ypred, ytrue)
            training_losses[e]+=np.sum(loss)
            training_errors[e]+=np.sum(np.argmax(ypred, axis=1)!=np.argmax(ytrue, axis=1))
        mlp.eval()
        for b in range(0, len(valx), batch_size):
            mlp.zero_grads()
            ypred=mlp.forward(valx[b:b+batch_size])
            ytrue=valy[b:b+batch_size]
            sce=SoftmaxCrossEntropy()
            loss=sce.forward(ypred, ytrue)
            validation_losses[e]+=np.sum(loss)
            validation_errors[e]+=np.sum(np.argmax(ypred, axis=1)!=np.argmax(ytrue, axis=1))
            # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...
        training_losses[e]/=trainx.shape[0]
        training_errors[e]/=trainx.shape[0]
        validation_losses[e]/=valx.shape[0]
        validation_errors[e]/=valx.shape[0]
    # Cleanup ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

    #raise NotImplemented
