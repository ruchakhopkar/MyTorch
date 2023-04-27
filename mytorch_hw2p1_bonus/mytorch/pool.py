import numpy as np

class MaxPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        batch_size,in_channel,input_width, input_height=x.shape
        output_width=int((input_width-self.kernel)/self.stride+1)
        output_height=int((input_height-self.kernel)/self.stride+1)
        out=np.zeros((batch_size, in_channel, output_width, output_height))
        self.pidx=np.zeros((batch_size, in_channel, output_width, output_height))
        self.qidx=np.zeros((batch_size, in_channel, output_width, output_height))
        self.input_height=input_height
        self.input_width=input_width
        for i in range(batch_size):
            for j in range(in_channel):
                p=0
                for m in range(0,input_width-self.kernel, self.stride):
                    q=0
                    for n in range(0, input_height-self.kernel, self.stride):
                        a=x[i, j, m:m+self.kernel, n:n+self.kernel]
                        r,s=np.unravel_index(a.argmax(), a.shape)
                        
                        self.pidx[i,j,p,q]=int(r)
                        self.qidx[i,j,p,q]=int(s)
                        out[i,j,p,q]=x[i,j,r+m, s+n]
                        
                        q+=1
                    p+=1
        return out
    
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, out_channel, output_width, output_height=delta.shape
        dx=np.zeros((batch_size, out_channel, self.input_width, self.input_height))
        for i in range(batch_size):
            for j in range(out_channel):
                for m in range(0,output_width):
                    p=m*self.stride
                    for n in range(0, output_height):
                        q=n*self.stride
                        dx[i,j,int(p+self.pidx[i,j,m,n]), int(q+self.qidx[i,j,m,n])]=delta[i,j,m,n]
        return dx     
                
        #raise NotImplementedError

class MeanPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        batch_size,in_channel,input_width, input_height=x.shape
        output_width=int((input_width-self.kernel)/self.stride+1)
        output_height=int((input_height-self.kernel)/self.stride+1)
        out=np.zeros((batch_size, in_channel, output_width, output_height))
        self.pidx=np.zeros((batch_size, in_channel, output_width, output_height))
        self.qidx=np.zeros((batch_size, in_channel, output_width, output_height))
        self.input_height=input_height
        self.input_width=input_width
        for i in range(batch_size):
            for j in range(in_channel):
                p=0
                for m in range(0,input_width-self.kernel, self.stride):
                    q=0
                    for n in range(0, input_height-self.kernel, self.stride):
                        a=x[i, j, m:m+self.kernel, n:n+self.kernel]
                        out[i,j,p,q]=np.mean(a)
                        
                        q+=1
                    p+=1
        return out
        

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, out_channel, output_width, output_height=delta.shape
        dx=np.zeros((batch_size, out_channel, self.input_width, self.input_height))
        for i in range(batch_size):
            for j in range(out_channel):
                for m in range(0,output_width):
                    p=m*self.stride
                    for n in range(0, output_height):
                        q=n*self.stride
                        dx[i,j,p:p+self.kernel, q:q+self.kernel]+=delta[i,j,m,n]/self.kernel**2
        return dx 
        raise NotImplementedError
