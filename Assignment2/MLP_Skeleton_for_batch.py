# -*- coding: utf-8 -*-
"""
Created on Sat 02/05/2016 22:05:52
@author: Kaibo Liu
"""


from __future__ import division
from __future__ import print_function

import sys

import cPickle
import numpy as np

import random
import time 
from math import exp, log, e
from copy import deepcopy


import normalization as nor

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    # vector elements in this class are stored in matrix[m,1] not ndarray [m,]
    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.d = input_dims
        self.m = hidden_units
        self.W = np.random.uniform(-1,1,(self.d, self.m))/10    # W = [d,m], matrix even m==1
        self.b = np.random.uniform(-1,1,(self.m))/10
        self.layer_i = []
        self.layer_o = 0
        self.back_w, self.back_b, self.back_x = 0,0,0
        self.dw, self.db = 0,0
    def forward(self, x):               # x is [d,]
    # DEFINE forward function
        self.dw, self.db = np.zeros((self.d, self.m)), np.zeros(self.m)
        self.layer_i = x
        self.layer_o = np.dot(self.W.T,self.layer_i.reshape(-1,1)).reshape(-1) + self.b    # [m,batch] = [d,m].T*[d,batch]+[m,1], trans matrix back to ndarray
        if (self.layer_o.shape[0] == 1):
            self.layer_o = self.layer_o[0]
        #return self.layer_o   
    def backward(
        self, 
        grad_output, 
    ):
    # DEFINE backward function
        #self.back_w = np.dot(self.layer_i,grad_output.T)    # [d,batch]·[m,batch].T->[d,m]
        #self.back_w = self.layer_i * grad_output            # [d,batch]*[1,batch]->[d,batch], different from W[d,batch]
        ##### batch have nothing to do with W,b, after mean, W and b keep their dimensions, while x is extended 1 dimension
        self.back_w = np.dot(self.layer_i.reshape(self.d,1),grad_output.reshape(1,self.m))  # [d,m] = [d,1]·[1,m]
        self.back_b = grad_output                            # [m,]
        self.back_x = np.dot(self.W, grad_output.reshape(self.m,1)).reshape(-1)             # [d,m]·[m,1], only used for l2

        #self.dw = -learning_rate * self.back_w + momentum * self.dw     # [d,m]
        #self.db = -learning_rate * self.back_b + momentum * self.db     # [m,1]
        #self.W += self.dw
        #self.b += self.db
    def update(self,pw,pb,learning_rate=0.0,momentum=0.0,l2_penalty=0.0):
        self.dw = -learning_rate * pw + momentum * self.dw     # [d,m]
        self.db = -learning_rate * pb + momentum * self.db     # [m,1]
        self.W += self.dw
        self.b += self.db
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def __init__(self, x=0):
        self.layer_i = x
        self.layer_o = []
        self.back = 0
    def forward(self, x):               # x is [m,]
    # DEFINE forward function
        self.layer_i = x                # self.layer_i is [m,]
        self.layer_o = deepcopy(x)      # deepcopy won't change the value of x
        self.layer_o[self.layer_o < 0] = 0
        #return self.layer_o

    def backward(
    # DEFINE backward function
        self,    
        grad_output,
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        y = deepcopy(self.layer_i)      # grad_output is [m,batch]
        y[y > 0] = 1
        y[y == 0] = 0.5
        y[y < 0] = 0
        self.back = grad_output * y     # [m,] = [m,] * [m,] element wise
        #return y
    
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self, x=0):
        self.layer_i = x
        self.layer_o = 0
        self.label = 0
        self.loss = 0.0
        self.back = 0
    def forward(self, x,y):
        # DEFINE forward function
        #if (x.shape[0] == 1):
        #    x = x[0]
        self.layer_i = x
        p = 1/(1+e ** (-x)) 
        if (p >= 1):
            print ('p >= 1')
        elif (p <= 0):
            print ('p <= 0')

        self.loss = -y * x - log(1-p)   # =-y * np.log(p) - (1-y)*np.log(1-p)
        self.layer_o = p        
        self.label = y

    def backward(
        self, 
        #grad_output, 
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0
    ):
        # DEFINE backward function
        self.back = self.layer_o-self.label 
# ADD other operations and data entries in SigmoidCrossEntropy if needed



class network(object):
    def __init__(self,input_dims, hidden_units):
        self.l1 = LinearTransform(input_dims, hidden_units)
        self.l2 = ReLU()
        self.l3 = LinearTransform(hidden_units, 1)
        self.l4 = SigmoidCrossEntropy()
    def forward(self,x,label):
        self.l1.forward(x)
        self.l2.forward(self.l1.layer_o)
        self.l3.forward(self.l2.layer_o)
        self.l4.forward(self.l3.layer_o,label)
    def backward(self):
        self.l4.backward()
        self.l3.backward(self.l4.back)
        self.l2.backward(self.l3.back_x)
        self.l1.backward(self.l2.back)
    def update(self,pw1,pb1,pw2,pb2,learning_rate=0.0, momentum=0.0,l2_penalty=0.0):
        self.l1.update(pw1,pb1,learning_rate, momentum)
        self.l3.update(pw2,pb2,learning_rate, momentum)
    def evaluate(self):
        predic = 0
        if (n.l4.layer_o >= 0.5):
            predic = 1
        return np.absolute(predic-n.l4.label)


    #def update(self):
    #    self.l1.W = self.l1.W +self.l1.dw

def RunMain():
    '''
    # test forward for all layers
    x = np.array([1,2,3,4,5,6]).reshape(3,2)
    print('x',x,'\n',x.shape[0],'examples')
    
    n0 = network(2,4)
    #print('LT1 W,b,d,m\n',n0.l1.W,'\n\n',n0.l1.b,'\n',n0.l1.d,n0.l1.m)
    #print('LT2 W,b,d,m\n',n0.l3.W,'\n\n',n0.l3.b,'\n',n0.l3.d,n0.l3.m)
    #print('LT1 W,b,d,m\n',n0.l1.W,'\n\n',n0.l1.b,'\n',n0.l1.d,n0.l1.m)
    n0.forward(x.T,np.array([1,0,1]),3)
    print (n0.l1.W)
    n0.backward(0.1,0.8)
    print (n0.l1.W)
    z0 = n0.l1.layer_o
    z1 = n0.l2.layer_o
    z2 = n0.l3.layer_o
    p  = n0.l4.layer_o
    E  = n0.l4.loss

    print('output_LT_1 ',n0.l1.layer_o)
    print('output_ReLU ',n0.l2.layer_o)
    print('output_LT_2 ',n0.l3.layer_o)
    print('output_sig ',n0.l4.layer_o)
    print('output_loss ',n0.l4.loss)
        
    print('backout_sigXe ',n0.l4.back, n0.l4.back.shape)
    print('backout_LT_2 ',n0.l3.back_x)
    print('backout_ReLU ',n0.l2.back)
    print('backout_LT_1 ',n0.l1.back_x)
    
    n0.forward(x.T,np.array([1,0,1]),3)
    n0.backward(0.1,0.8)
    print('output_loss ',n0.l4.loss)
    n0.forward(x.T,np.array([1,0,1]),3)
    n0.backward(0.1,0.8)
    print('output_loss ',n0.l4.loss)
    n0.forward(x.T,np.array([1,0,1]),3)
    n0.backward(0.1,0.8)
    print('output_loss ',n0.l4.loss)
    '''
    x = np.array([1,2,3,4,5,6]).reshape(2,-1)
    y = np.array([2,3,-4,5,-6,7])
    y = y.reshape(-1)
    k = 1/(1+e ** (-y))
    l = np.log(1-k)
    print (k,'\n',1-k,'\n',l)
    print(y*l)
    z = x[:,:-1]
    a = np.array([])
    
if __name__ == '__main__':

    #RunMain()
    
    t0 = float(time.clock())
    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
    t1 = float(time.clock())
    print ('Loading time is %.4f s. \n' % (t1-t0))
    t0 = t1

    train_y = data['train_labels']
    #test_x = data['test_data']
    test_y = data['test_labels']

    train_x, test_x= nor.Normalization(data['train_data'],data['test_data'])
    
    num_examples, input_dims = train_x.shape
    num_test = test_x.shape[0]
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 1000
    size_batch = int(num_examples / num_batches)
    hidden_units = 100              # tune
    lr = 0.01
    mu = 0.8                        # fix momentum to 0.8
    #mlp = MLP(input_dims, hidden_units)
    n = network(input_dims,hidden_units)
    
    for epoch in xrange(num_epochs):
    # INSERT YOUR CODE FOR EACH EPOCH HERE
        train_loss = 0.0
        error = 0
        start_line = 0
        for b in xrange(num_batches):
            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss
            total_loss = 0.0
            pw1, pb1, pw2, pb2 = 0.0, 0.0, 0.0, 0.0
            for i in xrange(start_line,(start_line+size_batch)):
                x = train_x[i]
                y = train_y[i][0]
                n.forward(x,y)
                n.backward()
                #print(n.l3.layer_o, n.l4.layer_o, n.l4.loss)
                pw1 += n.l1.back_w
                pb1 += n.l1.back_b

                pw2 += n.l3.back_w
                pb2 += n.l3.back_b

                total_loss += n.l4.loss
                error += n.evaluate()

            pw1 = pw1 / size_batch
            pb1 = pb1 / size_batch
            pw2 = pw2 / size_batch
            pb2 = pb2 / size_batch
            n.update(pw1,pb1,pw2,pb2,lr,mu)

            start_line += size_batch
            train_loss += total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss/size_batch,
                ),
                end='',
            )
            sys.stdout.flush()
        
        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        train_accuracy = 1 - error*1.0/num_examples
        train_loss = train_loss*1.0/num_examples
        #train_loss = train_loss
        error = 0
        test_loss = 0.0
        for i in xrange(num_test):
            x = test_x[i]
            y = test_y[i][0]
            n.forward(x,y)
            test_loss += n.l4.loss
            error += n.evaluate()
        test_accuracy = 1 - error*1.0/num_test
        test_loss = test_loss*1.0/num_test
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))
        
    t1 = float(time.clock())
    print ('Running time is %.4f s. \n' % (t1-t0))
