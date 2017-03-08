# -*- coding: utf-8 -*-
"""
CS519 Deep Learning, Assignment 2
@author: Kaibo Liu
"""
'''
should normalize every xi in all examples!!!Because x10 and x1d may have different unit,
but x10 and xN0 are in same unit. In conclusion, we should normalize the column vector of train data
the [0,1] normalization (also known as min-max) and the z-score normalization are two of the most widely used.
'''
import numpy as np

def Normalization(trainX,testX):        #[n,d] [n,1]

    Xmean  = np.mean(trainX,axis=0).reshape(1,-1)     #[1,d]
    Xstd   = np.std(trainX,axis=0).reshape(1,-1)      #[1,d]
    train1 = (trainX - Xmean) / Xstd
    test1  = (testX - Xmean) / Xstd
 
    return train1,test1