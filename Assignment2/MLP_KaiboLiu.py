# -*- coding: utf-8 -*-
"""
Created on Sat 02/05/2016 22:05:52
@author: Kaibo Liu
"""


from __future__ import division
from __future__ import print_function

import sys
import os

import cPickle
import numpy as np

import random
import time
from math import exp, log, e
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import normalization as nor



def get_colour(color):
    '''
    b---blue   c---cyan  g---green    k----black
    m---magenta r---red  w---white    y----yellow
'''
    color_set = ['r--','b--','m--','g--','c--','k--','y--']
    return color_set[color % 7]

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    # vector elements in this class are stored in matrix[m,1] not ndarray [m,]
    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.d = input_dims
        self.m = hidden_units
        self.batch = 0
        self.W = np.random.uniform(-1,1,(self.d, self.m))/10   # W = [d,m], matrix even m==1
        self.b = np.random.uniform(-1,1,(self.m,1))/10
        #self.W = np.random.uniform(0.5,0.5,(self.d, self.m))    # W = [d,m], matrix even m==1
        #self.b = np.random.uniform(0.6,0.6,(self.m, 1))
        self.layer_i = []
        self.layer_o = 0
        self.back_w, self.back_b, self.back_x = 0,0,0
        self.dw, self.db = 0,0
    def forward(self, x, size_batch):               # x is [d,batch]
    # DEFINE forward function
        self.batch = size_batch
        self.dw, self.db = np.zeros((self.d, self.m)), np.zeros((self.m,1))
        self.layer_i = x.reshape(self.d,self.batch)
        self.layer_o = np.dot(self.W.T,self.layer_i) + self.b    # [m,batch] = [d,m].T*[d,batch]+[m,1], trans matrix back to ndarray
        #return self.layer_o
    def backward(
        self,
        grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
    ):
    # DEFINE backward function
    ##### batch have nothing to do with W,b, after mean, W and b keep their dimensions, while x is extended 1 dimension
        self.back_w = np.zeros((self.d, self.m))
        for j in xrange(self.batch):
            self.back_w += np.dot(self.layer_i[:,j].reshape(self.d,1),grad_output[:,j].reshape(1,self.m))  #切片后[d,1]·[1,m]，为每个example的back
        self.back_w = self.back_w / self.batch

        self.back_b = np.mean(grad_output,axis = 1).reshape(-1,1)       # [m,batch]-->[m,]-->[m,1]
        self.back_x = np.dot(self.W, grad_output)           # [d,m]·[m,batch], only used for l2
        self.dw = momentum * self.dw - learning_rate * self.back_w    # [d,m]
        self.db = momentum * self.db - learning_rate * self.back_b    # [m,1]
        self.W += self.dw
        self.b += self.db

# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def __init__(self, x=0):
        self.layer_i = x
        self.layer_o = []
        self.back = 0
    def forward(self, x):               # x is [m,batch]
    # DEFINE forward function
        self.layer_i = x                # self.layer_i is [m,batch]
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
        self.back = grad_output * y     # [m,batch] = [m,batch] * [m,batch] element wise
        #return y

# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self, x=0):
        self.layer_i = x
        self.layer_o = 0
        self.label = 0
        self.loss = 0
        self.back = 0
    def forward(self, x,y):
        # DEFINE forward function
        x = x.reshape(-1)       # [1,batch]-->[batch,]
        y = y.reshape(-1)       # [batch,1]-->[batch,]

        self.layer_i = x
        temp_exp = e ** (-np.absolute(x))   # exp(-|x|)
        self.layer_o = np.where(x >= 0,1/(1+temp_exp),temp_exp/(1+temp_exp))   # p [batch,]
                # =p [batch,]
        #p[p >= 0] = 1/(1+temp_exp)          # p [batch,]
        #p[p < 0] = temp_exp/(1+temp_exp)    # p [batch,]
        #p = 1/(1+e ** (-x))     # p [batch,]
        '''
        for pi in p:
            if (pi >= 1):
                print ('p >= 1')
            elif (pi <= 0):
                print ('p <= 0')
        '''
        #self.loss = -y * np.log(p) - (1-y)*np.log(1-p) #E [1,batch]
        #self.loss = -y * x - np.log(1-p)   #E [batch,]
        self.loss = np.where(x<0,0,x) - y * x + np.log(1+temp_exp)   #E [batch,]
        self.label = y

    def backward(
        self,
        #grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0
    ):
        # DEFINE backward function
        self.back = (self.layer_o-self.label).reshape(1,-1)         # transfer [batch,] to [1,batch]

# ADD other operations and data entries in SigmoidCrossEntropy if needed



class network(object):
    def __init__(self,input_dims, hidden_units):
        self.l1 = LinearTransform(input_dims, hidden_units)
        self.l2 = ReLU()
        self.l3 = LinearTransform(hidden_units, 1)
        self.l4 = SigmoidCrossEntropy()
    def forward(self,x,label,size_batch):
        self.l1.forward(x, size_batch)
        self.l2.forward(self.l1.layer_o)
        self.l3.forward(self.l2.layer_o, size_batch)
        self.l4.forward(self.l3.layer_o,label)
    def backward(self,learning_rate=0.0, momentum=0.0):
        self.l4.backward()
        self.l3.backward(self.l4.back,learning_rate, momentum,l2_penalty=0.0)
        self.l2.backward(self.l3.back_x)
        self.l1.backward(self.l2.back,learning_rate, momentum,l2_penalty=0.0)
    def evaluate(self):
        predic = deepcopy(self.l4.layer_o)
        predic[predic >= 0.5] = 1
        predic[predic < 0.5]  = 0
        return np.sum(np.absolute(predic-self.l4.label))

    #def update(self):
    #    self.l1.W = self.l1.W +self.l1.dw

def RunMain(x,train_y,
            test_x,test_y,
            num_epochs,
            num_batches,
            hidden_units,
            lr,mu,
            ):

    num_examples, input_dims = x.shape
    num_test = test_x.shape[0]
    size_batch = int(num_examples / num_batches)
    n = network(input_dims,hidden_units)

    acc1, acc2, loss1, loss2 = [],[],[],[]
    for epoch in xrange(num_epochs):
    # INSERT YOUR CODE FOR EACH EPOCH HERE
        train_loss = 0.0
        start_line = 0
        for b in xrange(num_batches):
            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss
            total_loss = 0.0
            batch_x = x[start_line:(start_line+size_batch),:]
            batch_y = train_y[start_line:(start_line+size_batch)]
            start_line += size_batch
            n.forward(batch_x.T,batch_y,size_batch)
            n.backward(lr,mu)
            total_loss = np.sum(n.l4.loss)
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
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy

        n.forward(x.T,train_y,num_examples)  # this batch is all the enteties/lines in train
        train_loss = np.sum(n.l4.loss)/num_examples
        error = n.evaluate()
        train_accuracy = 1 - error*1.0/num_examples

        n.forward(test_x.T,test_y,num_test)  # this batch is all the enteties/lines in test
        test_loss = np.sum(n.l4.loss)/num_test
        error = n.evaluate()
        test_accuracy = 1 - error*1.0/num_test

        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))
        acc1.append(train_accuracy)     # [num_epochs,1]
        acc2.append(test_accuracy)      # [num_epochs,1]
        loss1.append(train_loss)        # [num_epochs,1]
        loss2.append(test_loss)         # [num_epochs,1]
    return acc1,acc2,loss1,loss2



def WriteData(acc_train_mat, acc_test_mat,
              loss_train_mat, loss_test_mat,
              num_batches,hidden_units,lr,
              num_epochs, tune_name, tune_list):
    #Acc againts epoch with different tune attributes
    color = 0
    for i in xrange(len(tune_list)):
        plt.figure(1)
        plt.plot(range(1,num_epochs+1),acc_train_mat[i], get_colour(color),label="%s %s" %(tune_name,str(tune_list[i])))
        plt.xlim(0, num_epochs)
        plt.ylim(0.65, 1)
        plt.xlabel('epoch')
        plt.ylabel('Train Accuracy')
        plt.legend(loc='upper left')
        plt.grid(True)

        plt.figure(2)
        plt.plot(range(1,num_epochs+1),acc_test_mat[i], get_colour(color),label="%s %s" %(tune_name,str(tune_list[i])))
        plt.xlim(0, num_epochs)
        plt.ylim(0.6, 0.85)
        plt.xlabel('epoch')
        plt.ylabel('Test Accuracy')
        plt.legend(loc='upper left')
        plt.grid(True)

        plt.figure(3)
        plt.plot(range(1,num_epochs+1),loss_train_mat[i], get_colour(color),label="%s %s" %(tune_name,str(tune_list[i])))
        plt.xlim(0, num_epochs)
        #plt.ylim(200, 2000)
        plt.xlabel('epoch')
        plt.ylabel('Train Loss')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.figure(4)
        plt.plot(range(1,num_epochs+1),loss_test_mat[i], get_colour(color),label="%s %s" %(tune_name,str(tune_list[i])))
        plt.xlim(0, num_epochs)
        #plt.ylim(200, 2000)
        plt.xlabel('epoch')
        plt.ylabel('Test Loss')
        plt.legend(loc='upper right')
        plt.grid(True)

        if (tune_name == 'hidden_units'):
            hidden_units = tune_list[i]
        elif (tune_name == 'learning_rate'):
            lr = tune_list[i]
        else:
            num_batches = tune_list[i]
        plt.figure()
        plt.plot(range(1,num_epochs+1),acc_train_mat[i], get_colour(0),label="train")
        plt.plot(range(1,num_epochs+1),acc_test_mat[i], get_colour(1),label="test")
        plt.xlim(0, num_epochs)
        plt.ylim(0.65, 1)
        plt.title('Accuracy with %d hidden units, %s learning rate, %d batches' %(hidden_units,str(lr),num_batches))
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(newdir+"Acc_hid_%d+lr_%s+mb_%d.png" %(hidden_units,str(lr),num_batches))
        color += 1

    # end of tune, save results

    if (tune_name == 'hidden_units'):
        plt.figure(1)
        plt.savefig(newdir+"TrainAcc_tune_hidden+lr_%s+mb_%d.png" %(str(lr),num_batches))
        plt.figure(2)
        plt.savefig(newdir+"TestAcc_tune_hidden+lr_%s+mb_%d.png" %(str(lr),num_batches))
        plt.figure(3)
        plt.savefig(newdir+"TrainLoss_tune_hidden+lr_%s+mb_%d.png" %(str(lr),num_batches))
        plt.figure(4)
        plt.savefig(newdir+"TestLoss_tune_hidden+lr_%s+mb_%d.png" %(str(lr),num_batches))

        np.savetxt(newdir+"TrainAcc_tune_hidden+lr_%s+mb_%d.csv" %(str(lr),num_batches), acc_train_mat, delimiter = ',')
        np.savetxt(newdir+"TestAcc_tune_hidden+lr_%s+mb_%d.csv" %(str(lr),num_batches), acc_test_mat, delimiter = ',')
        np.savetxt(newdir+"TrainLoss_tune_hidden+lr_%s+mb_%d.csv" %(str(lr),num_batches), loss_train_mat, delimiter = ',')
        np.savetxt(newdir+"TestLoss_tune_hidden+lr_%s+mb_%d.csv" %(str(lr),num_batches), loss_test_mat, delimiter = ',')
    elif (tune_name == 'learning_rate'):
        plt.figure(1)
        plt.savefig(newdir+"TrainAcc_tune_lr+hidden_%d+mb_%d.png" %(hidden_units,num_batches))
        plt.figure(2)
        plt.savefig(newdir+"TestAcc_tune_lr+hidden_%d+mb_%d.png" %(hidden_units,num_batches))
        plt.figure(3)
        plt.savefig(newdir+"TrainLoss_tune_lr+hidden_%d+mb_%d.png" %(hidden_units,num_batches))
        plt.figure(4)
        plt.savefig(newdir+"TestLoss_tune_lr+hidden_%d+mb_%d.png" %(hidden_units,num_batches))

        np.savetxt(newdir+"TrainAcc_tune_lr+hidden_%d+mb_%d.csv" %(hidden_units,num_batches), acc_train_mat, delimiter = ',')
        np.savetxt(newdir+"TestAcc_tune_lr+hidden_%d+mb_%d.csv" %(hidden_units,num_batches), acc_test_mat, delimiter = ',')
        np.savetxt(newdir+"TrainLoss_tune_lr+hidden_%d+mb_%d.csv" %(hidden_units,num_batches), loss_train_mat, delimiter = ',')
        np.savetxt(newdir+"TestLoss_tune_lr+hidden_%d+mb_%d.csv" %(hidden_units,num_batches), loss_test_mat, delimiter = ',')
    else:
        plt.figure(1)
        plt.savefig(newdir+"TrainAcc_tune_batch+lr_%s+hidden_%d.png" %(str(lr),hidden_units))
        plt.figure(2)
        plt.savefig(newdir+"TestAcc_tune_batch+lr_%s+hidden_%d.png" %(str(lr),hidden_units))
        plt.figure(3)
        plt.savefig(newdir+"TrainLoss_tune_batch+lr_%s+hidden_%d.png" %(str(lr),hidden_units))
        plt.figure(4)
        plt.savefig(newdir+"TestLoss_tune_batch+lr_%s+hidden_%d.png" %(str(lr),hidden_units))

        np.savetxt(newdir+"TrainAcc_tune_batch+lr_%s+hidden_%d.csv" %(str(lr),hidden_units), acc_train_mat, delimiter = ',')
        np.savetxt(newdir+"TestAcc_tune_batch+lr_%s+hidden_%d.csv" %(str(lr),hidden_units), acc_test_mat, delimiter = ',')
        np.savetxt(newdir+"TrainLoss_tune_batch+lr_%s+hidden_%d.csv" %(str(lr),hidden_units), loss_train_mat, delimiter = ',')
        np.savetxt(newdir+"TestLoss_tune_batch+lr_%s+hidden_%d.csv" %(str(lr),hidden_units), loss_test_mat, delimiter = ',')
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()
    plt.figure(3)
    plt.clf()
    plt.figure(4)
    plt.clf()

if __name__ == '__main__':

    t0 = t000 = float(time.clock())
    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
    t1 = float(time.clock())
    print ('Loading time is %.4f s. \n' % (t1-t0))

    newdir = './result/result_%s' %(t1)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    newdir += '/'
    f = open(newdir+'/runtime.txt', 'w')
    #plt.figure()
    #plt.plot([1,2,3,4,3,5],'-r')
    #plt.show()


    x, test_x = nor.Normalization(data['train_data'],data['test_data'])
    train_y = data['train_labels']
    test_y = data['test_labels']

    num_epochs = 30

    num_batches = 1000
    #batch_list = [100]    # tune
    batch_list = [100,500,1000,2000]    # tune
    hidden_units = 100                  # fix learning when tuning other
    #hiddenU_list = [100,1000]        # tune
    hiddenU_list = [10,100,1000]        # tune
    lr = 0.001                           # fix learning when tuning other
    #lr_list = [0.001]       # tune learning_rate
    lr_list = [0.005,0.001,0.0005,0.0001]       # tune learning_rate
    mu = 0.8                            # fix momentum to 0.8

    ############ tune learning rate  ###########
    t1 = t00 = float(time.clock())
    acc_train_mat, acc_test_mat, loss_train_mat, loss_test_mat = [],[],[],[]
    for l_r in lr_list:
        t0 = t1
        acc_train,acc_test, loss_train, loss_test = RunMain(x,train_y,
                test_x,test_y,
                num_epochs,
                num_batches,
                hidden_units,
                l_r,mu
                )
        acc_train_mat.append(acc_train)
        acc_test_mat.append(acc_test)
        loss_train_mat.append(loss_train)
        loss_test_mat.append(loss_test)

        t1 = float(time.clock())
        para = 'hidden units: %d, num_batch: %d, learning rate: %.4f, momentum: %.1f' %(hidden_units,num_batches,l_r,mu)
        print (para)
        print ('Running time is %.4f s. \n' % (t1-t0))
        f.write(para+'\nRunning time for 1 epoch is %.4f s. \n\n' %((t1-t0)/num_epochs))
    t1 = float(time.clock())
    print ('Time for tuning learning rate is %.4f s. \n' % (t1-t00))
    f.write('Running time for tuning learning rate is %.4f s. \n\n\n\n' %(t1-t00))
    f.write('******************************\n')
    # end of tune, save results
    WriteData(acc_train_mat, acc_test_mat,
              loss_train_mat, loss_test_mat,
              num_batches,hidden_units,lr,
              num_epochs, 'learning_rate', lr_list)

    ############ tune hidden units  ###########
    t1 = t00 = float(time.clock())
    acc_train_mat, acc_test_mat, loss_train_mat, loss_test_mat = [],[],[],[]
    for hidden_u in hiddenU_list:
        t0 = t1
        acc_train,acc_test, loss_train, loss_test = RunMain(x,train_y,
                test_x,test_y,
                num_epochs,
                num_batches,
                hidden_u,
                lr,mu
                )
        acc_train_mat.append(acc_train)
        acc_test_mat.append(acc_test)
        loss_train_mat.append(loss_train)
        loss_test_mat.append(loss_test)

        t1 = float(time.clock())
        para = 'hidden units: %d, num_batch: %d, learning rate: %.4f, momentum: %.1f' %(hidden_u,num_batches,lr,mu)
        print (para)
        print ('Running time is %.4f s. \n' % (t1-t0))
        f.write(para+'\nRunning time for 1 epoch is %.4f s. \n\n' %((t1-t0)/num_epochs))
    t1 = float(time.clock())
    print ('Time for tuning hidden units is %.4f s. \n' % (t1-t00))
    f.write('Running time for tuning hidden units is %.4f s. \n\n\n\n' %(t1-t00))
    f.write('******************************\n')
    # end of tune, save results
    WriteData(acc_train_mat, acc_test_mat,
              loss_train_mat, loss_test_mat,
              num_batches,hidden_units,lr,
              num_epochs, 'hidden_units', hiddenU_list)

    ############ tune batches  ###########
    t1 = t00 = float(time.clock())
    acc_train_mat, acc_test_mat, loss_train_mat, loss_test_mat = [],[],[],[]
    for batch in batch_list:
        t0 = t1
        acc_train,acc_test, loss_train, loss_test = RunMain(x,train_y,
                test_x,test_y,
                num_epochs,
                batch,
                hidden_units,
                lr,mu
                )
        acc_train_mat.append(acc_train)
        acc_test_mat.append(acc_test)
        loss_train_mat.append(loss_train)
        loss_test_mat.append(loss_test)

        t1 = float(time.clock())
        para = 'hidden units: %d, num_batch: %d, learning rate: %.4f, momentum: %.1f' %(hidden_units,batch,lr,mu)
        print (para)
        print ('Running time is %.4f s. \n' % (t1-t0))
        f.write(para+'\nRunning time for 1 epoch is %.4f s. \n\n' %((t1-t0)/num_epochs))
    t1 = float(time.clock())
    print ('Time for tuning batches is %.4f s. \n' % (t1-t00))
    f.write('Running time for tuning batches is %.4f s. \n\n\n\n' %(t1-t00))
    f.write('******************************\n')
    # end of tune, save results
    WriteData(acc_train_mat, acc_test_mat,
              loss_train_mat, loss_test_mat,
              num_batches,hidden_units,lr,
              num_epochs, 'batches', batch_list)

    t1 = float(time.clock())
    print ('Total running time is %.4f s. \n' % (t1-t000))
    f.write('\nToatl running time is %.4f s. \n' %(t1-t000))
    f.close()
