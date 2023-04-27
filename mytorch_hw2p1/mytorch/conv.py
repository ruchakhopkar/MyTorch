# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out_size=int(((x.shape[2]-self.kernel_size)//self.stride))+1
        self.x=x
        self.T=x.shape[2]
        z=np.zeros((x.shape[0], self.out_channel, out_size))
        for i in range(self.x.shape[0]):
            for t in range(0, int(np.floor((x.shape[2]-self.kernel_size)/self.stride))+1):                
                out=np.multiply(self.W, x[i,:, self.stride*t:self.stride*t+self.kernel_size])
                out=np.sum(out, axis=tuple(range(1,len(self.x.shape))))
                out=out+self.b
                z[i,:,t]=+out
        return z
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        #raise NotImplemented

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        dx=np.zeros((delta.shape[0], self.in_channel, self.T))
        for inp in range(delta.shape[0]):
            for x in range(0, delta.shape[2]):
                for j in range(self.out_channel):
                    for i in range(self.in_channel):
                        for xdash in range(self.kernel_size):
                            self.dW[j,i,xdash]+=np.multiply(delta[inp,j,x], self.x[inp,i,self.stride*x+xdash])
                            dx[inp,i,self.stride*x+xdash]+=np.multiply(self.W[j,i,xdash], delta[inp,j,x])
        self.db=(np.sum(delta, axis=(0,2)))
        return dx
        #raise NotImplemented


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x=x
        output_width=int(((x.shape[2]-self.kernel_size)//self.stride))+1
        output_height=int(((x.shape[3]-self.kernel_size)//self.stride))+1
        z=np.zeros((x.shape[0], self.out_channel, output_width, output_height))
        for inp in range(x.shape[0]):
            for w in range(0, int(np.floor((x.shape[2]-self.kernel_size)/self.stride))+1):
                for y in range(0, int(np.floor((x.shape[3]-self.kernel_size)/self.stride))+1):
                    out=np.multiply(self.W, x[inp,:,self.stride*w:self.stride*w+self.kernel_size,\
                                              self.stride*y:self.stride*y+self.kernel_size])
                    out=np.sum(out, axis=tuple(range(1,len(self.x.shape))))
                    out=out+self.b
                    z[inp,:,w,y]=+out
        return z
        #raise NotImplementedError

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        dx=np.zeros((delta.shape[0], self.in_channel, self.x.shape[2], self.x.shape[3]))
        for inp in range(delta.shape[0]):
            for x in range(0, delta.shape[2]):
                for y in range(0,delta.shape[3]):
                    for j in range(self.out_channel):
                        for i in range(self.in_channel):
                            for xdash in range(self.kernel_size):
                                for ydash in range(self.kernel_size):
                                    self.dW[j,i,xdash,ydash]+=np.multiply(delta[inp,j,x,y], \
                                                                    self.x[inp,i,self.stride*x+xdash, \
                                                                                     self.stride*y+ydash])
                                    dx[inp,i,self.stride*x+xdash, self.stride*y+ydash]+=np.multiply\
                                            (self.W[j,i,xdash, ydash], delta[inp,j,x,y])
        self.db=(np.sum(delta, axis=(0,2,3)))
        return dx
        #raise NotImplementedError
        


class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        self.kernel_dilated = (self.kernel_size-1)*(dilation-1)+self.kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        x=np.pad(x, ((0,0),(0,0),(self.padding, self.padding), (self.padding, self.padding)), \
                 mode='constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
        
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                kdash=0
                for k in range(0,self.W_dilated.shape[2],self.dilation):
                    ldash=0
                    for l in range(0,self.W_dilated.shape[3],self.dilation):
                        
                        self.W_dilated[i,j,k,l]=self.W[i,j,kdash,ldash]
                        ldash+=1
                    kdash+=1
        
        self.x=x
        output_width=int(((x.shape[2]-self.kernel_dilated)//self.stride))+1
        output_height=int(((x.shape[3]-self.kernel_dilated)//self.stride))+1
        z=np.zeros((x.shape[0], self.out_channel, output_width, output_height))
        for inp in range(x.shape[0]):
            for w in range(0, int(np.floor((x.shape[2]-self.kernel_dilated)/self.stride))+1):
                for y in range(0, int(np.floor((x.shape[3]-self.kernel_dilated)/self.stride))+1):
                    out=np.multiply(self.W_dilated, x[inp,:,self.stride*w:self.stride*w+self.kernel_dilated,\
                                              self.stride*y:self.stride*y+self.kernel_dilated])
                    out=np.sum(out, axis=tuple(range(1,len(self.x.shape))))
                    out=out+self.b
                    z[inp,:,w,y]=+out
        
        return z
        # TODO: padding x with self.padding parameter (HINT: use np.pad())
        
        # TODO: do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        #       HINT: for loop to get self.W_dilated

        # TODO: regular forward, just like Conv2d().forward()
        #raise NotImplementedError


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # TODO: main part is like Conv2d().backward(). The only difference are: we get padded input and dilated kernel
        #       for whole process while we only need original part of input and kernel for backpropagation.
        #       Please refer to writeup for more details.
        dW_dilated=np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))
        self.db=(np.sum(delta, axis=(0,2,3)))
        dx=np.zeros((delta.shape[0], self.in_channel, self.x.shape[2], self.x.shape[3]))
        for inp in range(delta.shape[0]):
            for x in range(0, delta.shape[2]):
                for y in range(0,delta.shape[3]):
                    for j in range(self.out_channel):
                        for i in range(self.in_channel):
                            for xdash in range(self.kernel_dilated):
                                for ydash in range(self.kernel_dilated):
                                    dx[inp,i,self.stride*x+xdash, self.stride*y+ydash]+=np.multiply\
                                            (self.W_dilated[j,i,xdash, ydash], delta[inp,j,x,y])
                                    dW_dilated[j,i,xdash,ydash]+=np.multiply(delta[inp,j,x,y], \
                                                                    self.x[inp,i,self.stride*x+xdash, \
                                                                                     self.stride*y+ydash])
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                kdash=0
                for k in range(0,self.W_dilated.shape[2],self.dilation):
                    ldash=0
                    for l in range(0,self.W_dilated.shape[3],self.dilation):
                        
                        self.dW[i,j,kdash,ldash]=dW_dilated[i,j,k,l]
                        ldash+=1
                    kdash+=1
        return dx[:,:,self.padding:-self.padding,self.padding:-self.padding]
        #raise NotImplementedError



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.in_width=x.shape[2]
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return np.reshape(x,(self.b, self.c*self.w))
        
        #raise NotImplemented

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return np.reshape(delta, (self.b,self.c, self.w))
        #raise NotImplemented

